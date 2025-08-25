"""
Conversor UDP para CSV em tempo real
Converte dados UDP (formato buffer) para formato CSV padrão OpenBCI em tempo real
Funciona em paralelo com o csv_data_logger
Aplica filtro Butterworth (0.5-50Hz) antes de salvar os dados
"""

import pandas as pd
import numpy as np
import os
import time
import threading
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
from .udp_receiver_BCI import UDPReceiver
from ..signal_processing.butter_filter import ButterworthFilter

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeUDPConverter:
    """
    Conversor UDP -> OpenBCI CSV em tempo real
    """
    
    def __init__(self, csv_filename: Optional[str] = None, host: str = 'localhost', port: int = 12345):
        """
        Inicializa o conversor em tempo real
        
        Args:
            csv_filename (str): Nome do arquivo CSV OpenBCI (se None, gera automaticamente)
            host (str): Host para receber dados UDP
            port (int): Porta para receber dados UDP
        """
        self.host = host
        self.port = port
        
        # Gerar nome do arquivo se não fornecido
        if csv_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.csv_filename = f"openbci_converted_{timestamp}.csv"
        else:
            self.csv_filename = csv_filename
            
        self.csv_filepath = os.path.join(os.getcwd(), self.csv_filename)
        
        # Receptor UDP
        self.udp_receiver = UDPReceiver(host=host, port=port)
        
        # Configurações
        self.is_converting = False
        self.sample_buffer = []
        self.buffer_size = 30  # Converter a cada 30 registros
        self.auto_flush_interval = 5  # Ou a cada 5 segundos
        self.last_flush_time = time.time()
        
        # Thread safety
        self.buffer_lock = threading.Lock()
        
        # Contador de amostras globais
        self.sample_index = 0
        self.total_samples_converted = 0
        self.conversion_errors = 0
        
        # Headers OpenBCI padrão
        self.openbci_headers = [
            'Sample Index', 'EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2', 'EXG Channel 3',
            'EXG Channel 4', 'EXG Channel 5', 'EXG Channel 6', 'EXG Channel 7', 'EXG Channel 8',
            'EXG Channel 9', 'EXG Channel 10', 'EXG Channel 11', 'EXG Channel 12', 'EXG Channel 13',
            'EXG Channel 14', 'EXG Channel 15', 'Accel Channel 0', 'Accel Channel 1', 'Accel Channel 2',
            'Other', 'Other.1', 'Other.2', 'Other.3', 'Other.4', 'Other.5', 'Other.6',
            'Analog Channel 0', 'Analog Channel 1', 'Analog Channel 2', 'Timestamp',
            'Other.7', 'Timestamp (Formatted)', 'Annotations'
        ]
        
        self.headers_written = False
        
        # Inicializar filtro Butterworth
        self.butter_filter = ButterworthFilter(
            lowcut=0.5,    # 0.5 Hz - remove artefatos de movimento 
            highcut=50.0,  # 50 Hz - remove ruído elétrico
            fs=125.0,      # 125 Hz - frequência de amostragem padrão OpenBCI
            order=6        # Ordem 6 - filtros em cascata (3+3) para estabilidade
        )
        
        logger.info(f"Filtro Butterworth inicializado: {self.butter_filter.get_filter_info()}")
        
    def _process_udp_data(self, data: Any):
        """
        Processa dados UDP recebidos e converte para formato OpenBCI
        
        Args:
            data: Dados recebidos do UDP
        """
        if not self.is_converting:
            return
            
        try:
            # Converter dados UDP para amostras OpenBCI
            converted_samples = self._convert_to_openbci_format(data)
            
            if converted_samples:
                # Adicionar ao buffer com thread safety
                with self.buffer_lock:
                    self.sample_buffer.extend(converted_samples)
                    self.total_samples_converted += len(converted_samples)
                
                logger.debug(f"Convertidas {len(converted_samples)} amostras para formato OpenBCI")
                
                # Verificar se precisa salvar
                self._check_flush_conditions()
                
        except Exception as e:
            self.conversion_errors += 1
            logger.error(f"Erro ao converter dados UDP: {e}")
            
    def _convert_to_openbci_format(self, data: Any) -> List[Dict]:
        """
        Converte dados UDP para formato OpenBCI padrão
        
        Args:
            data: Dados UDP recebidos
            
        Returns:
            Lista de amostras no formato OpenBCI
        """
        converted_samples = []
        
        try:
            # Se os dados são um dicionário JSON (formato OpenBCI GUI)
            if isinstance(data, dict):
                # Formato OpenBCI GUI: {'type': 'timeSeriesRaw', 'data': [[ch1_samples], [ch2_samples], ...]}
                if 'type' in data and data['type'] == 'timeSeriesRaw' and 'data' in data:
                    converted_samples = self._process_timeseries_raw(data['data'])
                    
                elif 'channels' in data:
                    # Formato: {'channels': {'Ch1': [valores], 'Ch2': [valores], ...}}
                    channels_data = data['channels']
                    converted_samples = self._process_channel_arrays(channels_data)
                    
                elif 'Ch1' in data:
                    # Formato direto: {'Ch1': [valores], 'Ch2': [valores], ...}
                    converted_samples = self._process_channel_arrays(data)
                    
                else:
                    # Formato de amostra única: {'Ch1': valor, 'Ch2': valor, ...}
                    converted_samples = self._process_single_sample(data)
                    
            elif isinstance(data, str):
                # Tentar parsear como JSON
                try:
                    json_data = json.loads(data)
                    return self._convert_to_openbci_format(json_data)
                except json.JSONDecodeError:
                    logger.warning(f"Dados UDP não são JSON válido: {data[:100]}...")
                    
        except Exception as e:
            logger.error(f"Erro na conversão para OpenBCI: {e}")
            
        return converted_samples
        
    def _process_channel_arrays(self, channels_data: Dict) -> List[Dict]:
        """
        Processa dados onde cada canal tem um array de valores
        """
        converted_samples = []
        
        try:
            # Extrair arrays de dados dos canais
            channel_arrays = []
            
            for ch in range(1, 17):  # Ch1 a Ch16
                ch_key = f'Ch{ch}'
                if ch_key in channels_data:
                    ch_data = channels_data[ch_key]
                    
                    if isinstance(ch_data, list):
                        channel_arrays.append(np.array(ch_data))
                    elif isinstance(ch_data, str):
                        # Parsear string como array
                        clean_data = ch_data.strip('[]"').replace('\n', ' ')
                        arr = np.fromstring(clean_data, sep=' ')
                        channel_arrays.append(arr)
                    else:
                        # Valor único
                        channel_arrays.append(np.array([ch_data]))
                else:
                    # Canal não encontrado, preencher com zeros
                    channel_arrays.append(np.array([0]))
            
            # Verificar se todos têm o mesmo tamanho
            array_lengths = [len(arr) for arr in channel_arrays]
            if len(set(array_lengths)) != 1:
                logger.warning(f"Tamanhos diferentes de arrays: {array_lengths}")
                return []
                
            buffer_size = array_lengths[0]
            
            # Aplicar filtro Butterworth aos dados antes de converter
            try:
                # Organizar dados em matriz (16 canais x N amostras)
                eeg_matrix = np.zeros((16, buffer_size))
                for ch in range(16):
                    eeg_matrix[ch, :] = channel_arrays[ch]
                
                # Aplicar filtro Butterworth
                filtered_eeg_matrix = self.butter_filter.apply_filter(eeg_matrix)
                
                # Atualizar arrays com dados filtrados
                for ch in range(16):
                    channel_arrays[ch] = filtered_eeg_matrix[ch, :]
                    
                logger.debug(f"Filtro Butterworth aplicado em {buffer_size} amostras, 16 canais")
                
            except Exception as filter_error:
                logger.warning(f"Erro ao aplicar filtro Butterworth: {filter_error}")
                # Continuar com dados sem filtrar em caso de erro
            
            # Converter cada amostra do buffer
            for sample_in_buffer in range(buffer_size):
                sample_data = {
                    'Sample Index': self.sample_index,
                    'EXG Channel 0': float(channel_arrays[0][sample_in_buffer]),
                    'EXG Channel 1': float(channel_arrays[1][sample_in_buffer]),
                    'EXG Channel 2': float(channel_arrays[2][sample_in_buffer]),
                    'EXG Channel 3': float(channel_arrays[3][sample_in_buffer]),
                    'EXG Channel 4': float(channel_arrays[4][sample_in_buffer]),
                    'EXG Channel 5': float(channel_arrays[5][sample_in_buffer]),
                    'EXG Channel 6': float(channel_arrays[6][sample_in_buffer]),
                    'EXG Channel 7': float(channel_arrays[7][sample_in_buffer]),
                    'EXG Channel 8': float(channel_arrays[8][sample_in_buffer]),
                    'EXG Channel 9': float(channel_arrays[9][sample_in_buffer]),
                    'EXG Channel 10': float(channel_arrays[10][sample_in_buffer]),
                    'EXG Channel 11': float(channel_arrays[11][sample_in_buffer]),
                    'EXG Channel 12': float(channel_arrays[12][sample_in_buffer]),
                    'EXG Channel 13': float(channel_arrays[13][sample_in_buffer]),
                    'EXG Channel 14': float(channel_arrays[14][sample_in_buffer]),
                    'EXG Channel 15': float(channel_arrays[15][sample_in_buffer]),
                    # Colunas padrão OpenBCI preenchidas com zeros
                    'Accel Channel 0': 0,
                    'Accel Channel 1': 0,
                    'Accel Channel 2': 0,
                    'Other': 0,
                    'Other.1': 0,
                    'Other.2': 0,
                    'Other.3': 0,
                    'Other.4': 0,
                    'Other.5': 0,
                    'Other.6': 0,
                    'Analog Channel 0': 0,
                    'Analog Channel 1': 0,
                    'Analog Channel 2': 0,
                    'Timestamp': self.sample_index,
                    'Other.7': 0,
                    'Timestamp (Formatted)': datetime.now().strftime('%H:%M:%S.%f')[:-3],
                    'Annotations': ''
                }
                
                converted_samples.append(sample_data)
                self.sample_index += 1
                
        except Exception as e:
            logger.error(f"Erro ao processar arrays de canais: {e}")
            
        return converted_samples
        
    def _process_single_sample(self, data: Dict) -> List[Dict]:
        """
        Processa dados onde cada canal tem um único valor
        """
        try:
            sample_data = {
                'Sample Index': self.sample_index,
                'Timestamp (Formatted)': datetime.now().strftime('%H:%M:%S.%f')[:-3],
                'Annotations': ''
            }
            
            # Extrair valores dos canais
            eeg_sample = np.zeros(16)
            for ch in range(16):
                ch_key = f'Ch{ch+1}'
                if ch_key in data:
                    eeg_sample[ch] = float(data[ch_key])
                else:
                    eeg_sample[ch] = 0.0
            
            # Aplicar filtro Butterworth para amostra única
            try:
                filtered_eeg_sample = self.butter_filter.apply_realtime_filter(eeg_sample)
                logger.debug("Filtro Butterworth aplicado em amostra única")
            except Exception as filter_error:
                logger.warning(f"Erro ao aplicar filtro em amostra única: {filter_error}")
                filtered_eeg_sample = eeg_sample
            
            # Preencher dados filtrados
            for ch in range(16):
                sample_data[f'EXG Channel {ch}'] = float(filtered_eeg_sample[ch])
            
            # Preencher outras colunas com zeros
            other_columns = [
                'Accel Channel 0', 'Accel Channel 1', 'Accel Channel 2',
                'Other', 'Other.1', 'Other.2', 'Other.3', 'Other.4', 'Other.5', 'Other.6',
                'Analog Channel 0', 'Analog Channel 1', 'Analog Channel 2',
                'Timestamp', 'Other.7'
            ]
            
            for col in other_columns:
                sample_data[col] = 0
                
            self.sample_index += 1
            return [sample_data]
            
        except Exception as e:
            logger.error(f"Erro ao processar amostra única: {e}")
            return []
            
    def _process_timeseries_raw(self, data_array: List[List]) -> List[Dict]:
        """
        Processa dados no formato timeSeriesRaw do OpenBCI GUI
        
        Args:
            data_array: Array 2D onde cada linha é um canal e cada coluna é uma amostra
                       [[ch1_sample1, ch1_sample2, ...], [ch2_sample1, ch2_sample2, ...], ...]
                       
        Returns:
            Lista de amostras no formato OpenBCI
        """
        converted_samples = []
        
        try:
            # data_array tem formato: [[canal0_amostras], [canal1_amostras], ...]
            if not data_array or len(data_array) == 0:
                return []
                
            # Determinar número de canais e amostras
            num_channels = len(data_array)
            num_samples = len(data_array[0]) if data_array[0] else 0
            
            # Verificar se todos os canais têm o mesmo número de amostras
            for i, channel_data in enumerate(data_array):
                if len(channel_data) != num_samples:
                    logger.warning(f"Canal {i} tem {len(channel_data)} amostras, esperado {num_samples}")
                    return []
            
            logger.debug(f"Processando {num_channels} canais x {num_samples} amostras")
            
            # Aplicar filtro Butterworth se há dados suficientes
            filtered_data_array = data_array.copy()
            try:
                if num_samples > 3 * self.butter_filter.order and num_channels >= 16:
                    # Organizar dados em matriz (16 canais x N amostras)
                    eeg_matrix = np.zeros((16, num_samples))
                    for ch in range(16):
                        if ch < num_channels:
                            eeg_matrix[ch, :] = data_array[ch]
                        else:
                            eeg_matrix[ch, :] = 0.0
                    
                    # Aplicar filtro
                    filtered_eeg_matrix = self.butter_filter.apply_filter(eeg_matrix)
                    
                    # Atualizar dados filtrados
                    filtered_data_array = []
                    for ch in range(num_channels):
                        if ch < 16:
                            filtered_data_array.append(filtered_eeg_matrix[ch, :].tolist())
                        else:
                            filtered_data_array.append(data_array[ch])
                    
                    logger.debug(f"Filtro Butterworth aplicado em timeSeriesRaw: {num_samples} amostras, {min(num_channels, 16)} canais")
                    
            except Exception as filter_error:
                logger.warning(f"Erro ao aplicar filtro em timeSeriesRaw: {filter_error}")
                filtered_data_array = data_array  # Usar dados originais
            
            # Converter cada amostra temporal
            for sample_idx in range(num_samples):
                sample_data = {
                    'Sample Index': self.sample_index,
                    'Timestamp': self.sample_index,
                    'Timestamp (Formatted)': datetime.now().strftime('%H:%M:%S.%f')[:-3],
                    'Annotations': ''
                }
                
                # Extrair valores de cada canal para esta amostra (usando dados filtrados)
                for ch_idx in range(16):  # OpenBCI padrão tem 16 canais
                    if ch_idx < len(filtered_data_array):
                        # Canal existe nos dados
                        sample_data[f'EXG Channel {ch_idx}'] = float(filtered_data_array[ch_idx][sample_idx])
                    else:
                        # Canal não existe, preencher com zero
                        sample_data[f'EXG Channel {ch_idx}'] = 0.0
                
                # Preencher outras colunas com zeros
                other_columns = [
                    'Accel Channel 0', 'Accel Channel 1', 'Accel Channel 2',
                    'Other', 'Other.1', 'Other.2', 'Other.3', 'Other.4', 'Other.5', 'Other.6',
                    'Analog Channel 0', 'Analog Channel 1', 'Analog Channel 2', 'Other.7'
                ]
                
                for col in other_columns:
                    sample_data[col] = 0
                
                converted_samples.append(sample_data)
                self.sample_index += 1
                
            logger.debug(f"Convertidas {len(converted_samples)} amostras do formato timeSeriesRaw")
            
        except Exception as e:
            logger.error(f"Erro ao processar timeSeriesRaw: {e}")
            
        return converted_samples
        
    def _check_flush_conditions(self):
        """
        Verifica se deve salvar o buffer no arquivo CSV
        """
        current_time = time.time()
        
        # Verificar condições sem lock prolongado
        with self.buffer_lock:
            buffer_size = len(self.sample_buffer)
            
        # Salvar se buffer atingiu tamanho máximo ou tempo limite
        if (buffer_size >= self.buffer_size or 
            current_time - self.last_flush_time >= self.auto_flush_interval):
            # Flush em thread separada para não bloquear captura
            threading.Thread(target=self._flush_to_csv, daemon=True).start()
            
    def _flush_to_csv(self):
        """
        Salva o buffer no arquivo CSV formato OpenBCI
        """
        # Copiar e limpar buffer atomicamente
        with self.buffer_lock:
            if not self.sample_buffer:
                return
            data_to_save = self.sample_buffer.copy()
            self.sample_buffer.clear()
            
        try:
            # Verificar se arquivo existe
            file_exists = os.path.exists(self.csv_filepath)
            
            # Se arquivo não existe, criar com cabeçalho OpenBCI
            if not file_exists or not self.headers_written:
                self._write_openbci_header()
                self.headers_written = True
                
            # Criar DataFrame e salvar
            df = pd.DataFrame(data_to_save, columns=self.openbci_headers)
            
            # Append ao arquivo
            df.to_csv(self.csv_filepath, mode='a', header=False, index=False)
            
            logger.info(f"✅ Salvos {len(data_to_save)} samples OpenBCI em {self.csv_filename}")
            
            # Atualizar timestamp
            self.last_flush_time = time.time()
            
        except Exception as e:
            logger.error(f"❌ Erro ao salvar CSV OpenBCI: {e}")
            # Em caso de erro, recolocar dados no buffer
            with self.buffer_lock:
                self.sample_buffer = data_to_save + self.sample_buffer
                
    def _write_openbci_header(self):
        """
        Escreve o cabeçalho padrão OpenBCI no arquivo
        """
        header_lines = [
            "%OpenBCI Raw EXG Data",
            "%Number of channels = 16", 
            "%Sample Rate = 125 Hz",
            "%Board = OpenBCI_GUI$BoardCytonSerialDaisy"
        ]
        
        with open(self.csv_filepath, 'w') as f:
            # Escreve o cabeçalho de comentários
            for line in header_lines:
                f.write(line + '\n')
            
            # Escreve o cabeçalho das colunas
            column_headers = ','.join(self.openbci_headers)
            f.write(column_headers + '\n')
                
    def start_converting(self):
        """
        Inicia a conversão em tempo real
        """
        if self.is_converting:
            logger.warning("Conversor já está ativo")
            return
            
        try:
            # Resetar estado do filtro para nova sessão
            self.butter_filter.reset_filter_state()
            logger.info("Estado do filtro Butterworth resetado para nova sessão")
            
            # Configurar callback no receptor UDP
            self.udp_receiver.set_callback(self._process_udp_data)
            
            # Iniciar receptor UDP
            self.udp_receiver.start()
            
            self.is_converting = True
            logger.info(f"Conversor OpenBCI iniciado. Arquivo: {self.csv_filepath}")
            
        except Exception as e:
            logger.error(f"Erro ao iniciar conversor: {e}")
            self.is_converting = False
            
    def stop_converting(self):
        """
        Para a conversão
        """
        if not self.is_converting:
            return
            
        logger.info("Parando conversor OpenBCI...")
        self.is_converting = False
        
        # Parar receptor UDP
        self.udp_receiver.stop()
        
        # Salvar dados restantes no buffer
        if self.sample_buffer:
            self._flush_final()
            
        logger.info("Conversor OpenBCI parado")
        
    def _flush_final(self):
        """
        Flush final garantindo que todos os dados sejam salvos
        """
        with self.buffer_lock:
            if not self.sample_buffer:
                return
            data_to_save = self.sample_buffer.copy()
            self.sample_buffer.clear()
            
        try:
            if not self.headers_written:
                self._write_openbci_header()
                
            df = pd.DataFrame(data_to_save, columns=self.openbci_headers)
            df.to_csv(self.csv_filepath, mode='a', header=False, index=False)
            
            logger.info(f"✅ Flush final: {len(data_to_save)} samples OpenBCI salvos")
            
        except Exception as e:
            logger.error(f"❌ Erro no flush final: {e}")
            
    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas do conversor
        """
        with self.buffer_lock:
            buffer_size = len(self.sample_buffer)
            
        return {
            'is_converting': self.is_converting,
            'csv_file': self.csv_filepath,
            'buffer_size': buffer_size,
            'total_udp_received': self.udp_receiver.get_data_count(),
            'total_samples_converted': self.total_samples_converted,
            'conversion_errors': self.conversion_errors,
            'current_sample_index': self.sample_index,
            'file_exists': os.path.exists(self.csv_filepath),
            'file_size_bytes': os.path.getsize(self.csv_filepath) if os.path.exists(self.csv_filepath) else 0,
            'last_save': datetime.fromtimestamp(self.last_flush_time).strftime("%H:%M:%S")
        }
        
    def force_save(self):
        """
        Força salvamento imediato
        """
        if self.sample_buffer:
            self._flush_to_csv()
            logger.info("Salvamento OpenBCI forçado concluído")
        else:
            logger.info("Nenhum dado no buffer para salvar")


def main():
    """
    Teste do conversor em tempo real
    """
    print("🔄 CONVERSOR UDP -> OpenBCI CSV EM TEMPO REAL")
    print("=" * 60)
    
    converter = RealTimeUDPConverter()
    
    try:
        converter.start_converting()
        
        print(f"🔄 Conversor rodando em localhost:12345")
        print(f"📁 Arquivo OpenBCI: {converter.csv_filename}")
        print("🛑 Pressione Ctrl+C para parar")
        print("=" * 60)
        
        counter = 0
        while True:
            time.sleep(3)
            counter += 3
            
            stats = converter.get_stats()
            print(f"[{counter:03d}s] 🔄 UDP: {stats['total_udp_received']:4d} | "
                  f"Samples: {stats['total_samples_converted']:5d} | "
                  f"Buffer: {stats['buffer_size']:2d} | "
                  f"Erros: {stats['conversion_errors']:2d} | "
                  f"Arquivo: {stats['file_size_bytes']:6d} bytes")
            
    except KeyboardInterrupt:
        print("\n🛑 Parando conversor...")
        converter.stop_converting()
        
        stats = converter.get_stats()
        print("📊 ESTATÍSTICAS FINAIS:")
        print(f"   • UDP recebidos: {stats['total_udp_received']}")
        print(f"   • Samples convertidos: {stats['total_samples_converted']}")
        print(f"   • Erros: {stats['conversion_errors']}")
        print(f"   • Arquivo: {stats['csv_file']}")
        print("✅ Conversor parado!")


if __name__ == "__main__":
    main()