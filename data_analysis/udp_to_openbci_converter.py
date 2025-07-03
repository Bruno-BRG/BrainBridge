"""
Conversor UDP para formato OpenBCI em tempo real.

Este módulo converte dados UDP (formato buffer) para o formato esperado pelo modelo BCI,
mantendo a estrutura de colunas do OpenBCI mas sem os headers desnecessários.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Optional, Tuple
import time

logger = logging.getLogger(__name__)

class UDPToOpenBCIConverter:
    """
    Conversor que transforma dados UDP em formato OpenBCI em tempo real.
    
    Funciona assim:
    1. Recebe dados UDP no formato buffer (16 canais x N amostras cada)
    2. Converte para formato OpenBCI (Sample Index, EXG Channel 0-15, ...)
    3. Retorna dados prontos para o modelo BCI
    """
    
    def __init__(self):
        self.sample_index = 0
        self.conversion_stats = {
            'total_buffers_processed': 0,
            'total_samples_converted': 0,
            'last_conversion_time': None,
            'errors': 0
        }
        
    def convert_udp_buffer_to_openbci(self, udp_data: List[List[float]], 
                                    global_sample_index: Optional[int] = None) -> pd.DataFrame:
        """
        Converte um buffer UDP para formato OpenBCI.
        
        Args:
            udp_data: Lista de 16 listas, cada uma com N amostras por canal
                     Formato: [[ch1_samples], [ch2_samples], ..., [ch16_samples]]
            global_sample_index: Índice global da primeira amostra (opcional)
            
        Returns:
            DataFrame com colunas: Sample Index, EXG Channel 0-15, Accel Channel 0-2, 
                                 Other, Other.1-6, Analog Channel 0-2, Timestamp, 
                                 Other.7, Timestamp (Formatted), Annotations
        """
        try:
            start_time = time.time()
            
            # Validar entrada
            if not isinstance(udp_data, list) or len(udp_data) != 16:
                raise ValueError(f"Esperado lista com 16 canais, recebido {len(udp_data) if isinstance(udp_data, list) else type(udp_data)}")
            
            # Verificar se todos os canais têm o mesmo número de amostras
            samples_per_channel = len(udp_data[0])
            for i, channel in enumerate(udp_data):
                if len(channel) != samples_per_channel:
                    raise ValueError(f"Canal {i} tem {len(channel)} amostras, esperado {samples_per_channel}")
            
            # Criar lista para armazenar todas as amostras convertidas
            converted_samples = []
            
            # Usar global_sample_index se fornecido, senão usar o contador interno
            start_index = global_sample_index if global_sample_index is not None else self.sample_index
            
            # Converter cada amostra do buffer
            for sample_idx in range(samples_per_channel):
                sample_data = {
                    # Índice da amostra
                    'Sample Index': start_index + sample_idx,
                    
                    # Dados EEG dos 16 canais (EXG Channel 0-15)
                    'EXG Channel 0': udp_data[0][sample_idx],
                    'EXG Channel 1': udp_data[1][sample_idx],
                    'EXG Channel 2': udp_data[2][sample_idx],
                    'EXG Channel 3': udp_data[3][sample_idx],
                    'EXG Channel 4': udp_data[4][sample_idx],
                    'EXG Channel 5': udp_data[5][sample_idx],
                    'EXG Channel 6': udp_data[6][sample_idx],
                    'EXG Channel 7': udp_data[7][sample_idx],
                    'EXG Channel 8': udp_data[8][sample_idx],
                    'EXG Channel 9': udp_data[9][sample_idx],
                    'EXG Channel 10': udp_data[10][sample_idx],
                    'EXG Channel 11': udp_data[11][sample_idx],
                    'EXG Channel 12': udp_data[12][sample_idx],
                    'EXG Channel 13': udp_data[13][sample_idx],
                    'EXG Channel 14': udp_data[14][sample_idx],
                    'EXG Channel 15': udp_data[15][sample_idx],
                    
                    # Colunas auxiliares (preenchidas com zeros como no formato original)
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
                    'Timestamp': 0,
                    'Other.7': 0,
                    'Timestamp (Formatted)': 0,
                    'Annotations': ''
                }
                
                converted_samples.append(sample_data)
            
            # Criar DataFrame
            df = pd.DataFrame(converted_samples)
            
            # Atualizar contador interno se não foi fornecido índice global
            if global_sample_index is None:
                self.sample_index += samples_per_channel
            
            # Atualizar estatísticas
            self.conversion_stats['total_buffers_processed'] += 1
            self.conversion_stats['total_samples_converted'] += samples_per_channel
            self.conversion_stats['last_conversion_time'] = time.time()
            
            conversion_time = time.time() - start_time
            logger.debug(f"✅ Buffer convertido: {samples_per_channel} amostras em {conversion_time*1000:.2f}ms")
            
            return df
            
        except Exception as e:
            self.conversion_stats['errors'] += 1
            logger.error(f"❌ Erro na conversão UDP->OpenBCI: {e}")
            raise
    
    def convert_numpy_buffer_to_openbci(self, numpy_buffer: np.ndarray, 
                                      global_sample_index: Optional[int] = None) -> pd.DataFrame:
        """
        Converte um buffer numpy (formato usado internamente) para OpenBCI.
        
        Args:
            numpy_buffer: Array numpy com shape (16, N) - 16 canais x N amostras
            global_sample_index: Índice global da primeira amostra (opcional)
            
        Returns:
            DataFrame no formato OpenBCI
        """
        try:
            # Validar shape
            if numpy_buffer.shape[0] != 16:
                raise ValueError(f"Esperado 16 canais, recebido {numpy_buffer.shape[0]}")
            
            # Converter para lista de listas
            udp_data = [numpy_buffer[i, :].tolist() for i in range(16)]
            
            # Usar a função principal de conversão
            return self.convert_udp_buffer_to_openbci(udp_data, global_sample_index)
            
        except Exception as e:
            logger.error(f"❌ Erro na conversão numpy->OpenBCI: {e}")
            raise
    
    def extract_eeg_channels_only(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extrai apenas os canais EEG do DataFrame OpenBCI.
        
        Args:
            df: DataFrame no formato OpenBCI
            
        Returns:
            Array numpy com shape (16, N) contendo apenas os dados EEG
        """
        try:
            eeg_columns = [f'EXG Channel {i}' for i in range(16)]
            eeg_data = df[eeg_columns].values.T  # Transpor para (16, N)
            return eeg_data.astype(np.float32)
            
        except Exception as e:
            logger.error(f"❌ Erro ao extrair canais EEG: {e}")
            raise
    
    def reset_sample_index(self, new_index: int = 0):
        """Reset do contador de amostras."""
        self.sample_index = new_index
        logger.info(f"🔄 Sample index resetado para {new_index}")
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas da conversão."""
        return self.conversion_stats.copy()
    
    def save_openbci_csv(self, df: pd.DataFrame, filename: str, include_headers: bool = False):
        """
        Salva DataFrame OpenBCI em arquivo CSV.
        
        Args:
            df: DataFrame no formato OpenBCI
            filename: Nome do arquivo
            include_headers: Se True, inclui headers OpenBCI no arquivo
        """
        try:
            if include_headers:
                # Salvar com headers OpenBCI
                with open(filename, 'w', newline='') as f:
                    f.write("%OpenBCI Raw EXG Data\n")
                    f.write("%Number of channels = 16\n")
                    f.write("%Sample Rate = 125 Hz\n")
                    f.write("%Board = OpenBCI_GUI$BoardCytonSerialDaisy\n")
                    df.to_csv(f, index=False)
            else:
                # Salvar apenas os dados
                df.to_csv(filename, index=False)
                
            logger.info(f"💾 Dados OpenBCI salvos: {filename}")
            
        except Exception as e:
            logger.error(f"❌ Erro ao salvar CSV: {e}")
            raise


# Funções de conveniência para uso direto
def convert_udp_to_openbci_format(udp_data: List[List[float]], 
                                global_sample_index: Optional[int] = None) -> pd.DataFrame:
    """
    Função de conveniência para conversão rápida UDP -> OpenBCI.
    
    Args:
        udp_data: Lista de 16 listas com amostras por canal
        global_sample_index: Índice global da primeira amostra
        
    Returns:
        DataFrame no formato OpenBCI
    """
    converter = UDPToOpenBCIConverter()
    return converter.convert_udp_buffer_to_openbci(udp_data, global_sample_index)


def convert_numpy_to_openbci_format(numpy_buffer: np.ndarray, 
                                  global_sample_index: Optional[int] = None) -> pd.DataFrame:
    """
    Função de conveniência para conversão rápida numpy -> OpenBCI.
    
    Args:
        numpy_buffer: Array numpy (16, N)
        global_sample_index: Índice global da primeira amostra
        
    Returns:
        DataFrame no formato OpenBCI
    """
    converter = UDPToOpenBCIConverter()
    return converter.convert_numpy_buffer_to_openbci(numpy_buffer, global_sample_index)


# Exemplo de uso
if __name__ == "__main__":
    # Exemplo com dados simulados
    logging.basicConfig(level=logging.INFO)
    
    # Simular dados UDP (16 canais x 5 amostras cada)
    udp_exemplo = [
        [14.88, 14.39, 10.91, 2.46, -4.35e-09],  # Canal 1
        [2.14, 6.36, 10.31, 6.00, -4.35e-09],   # Canal 2
        # ... mais 14 canais
    ]
    
    # Preencher com dados aleatórios para demonstração
    import random
    for i in range(len(udp_exemplo), 16):
        udp_exemplo.append([random.uniform(-50, 50) for _ in range(5)])
    
    # Converter
    converter = UDPToOpenBCIConverter()
    df_openbci = converter.convert_udp_buffer_to_openbci(udp_exemplo)
    
    print("✅ Conversão realizada!")
    print(f"📊 Shape do resultado: {df_openbci.shape}")
    print(f"🏛️ Colunas: {list(df_openbci.columns)}")
    print("\n📋 Primeiras 3 linhas:")
    print(df_openbci.head(3))
    
    # Extrair apenas dados EEG para o modelo
    eeg_only = converter.extract_eeg_channels_only(df_openbci)
    print(f"\n🧠 Dados EEG extraídos: {eeg_only.shape}")
    print(f"📊 Range: {eeg_only.min():.3f} a {eeg_only.max():.3f}")
    
    # Estatísticas
    stats = converter.get_stats()
    print(f"\n📈 Estatísticas: {stats}")
