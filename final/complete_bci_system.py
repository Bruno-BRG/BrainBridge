"""
Sistema BCI Completo - Captura e Predição em Tempo Real
Recebe dados UDP, salva em CSV, converte para formato OpenBCI, e faz predições em tempo real
"""

import sys
import time
import threading
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import argparse

# Adicionar pasta final ao path
sys.path.append(str(Path(__file__).parent))

from udp_receiver import UDPReceiver
from csv_data_logger import CSVDataLogger
from realtime_udp_converter import RealtimeUDPConverter
from realtime_bci_system import RealtimeBCIPredictor, default_prediction_callback

class CompleteBCISystem:
    """Sistema BCI completo com captura e predição"""
    
    def __init__(self, 
                 model_path: str,
                 save_raw_data: bool = True,
                 save_converted_data: bool = True,
                 enable_predictions: bool = True,
                 udp_host: str = 'localhost',
                 udp_port: int = 12345):
        
        self.model_path = Path(model_path)
        self.save_raw_data = save_raw_data
        self.save_converted_data = save_converted_data
        self.enable_predictions = enable_predictions
        self.udp_host = udp_host
        self.udp_port = udp_port
        
        # Componentes do sistema
        self.udp_receiver = None
        self.csv_logger = None
        self.udp_converter = None
        self.bci_predictor = None
        
        # Controle de execução
        self.running = False
        self.stats = {
            'total_packets': 0,
            'raw_samples_saved': 0,
            'converted_samples_saved': 0,
            'predictions_made': 0,
            'start_time': None
        }
        
        print(f"🚀 Sistema BCI Completo Inicializado")
        print(f"📂 Modelo: {self.model_path}")
        print(f"🌐 UDP: {udp_host}:{udp_port}")
        print(f"💾 Salvar dados brutos: {save_raw_data}")
        print(f"🔄 Salvar dados convertidos: {save_converted_data}")
        print(f"🧠 Predições habilitadas: {enable_predictions}")
    
    def _setup_components(self):
        """Configurar componentes do sistema"""
        try:
            # Configurar receptor UDP
            self.udp_receiver = UDPReceiver(host=self.udp_host, port=self.udp_port)
            
            # Configurar logger de dados brutos
            if self.save_raw_data:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                raw_filename = f"raw_data_bci_{timestamp}.csv"
                self.csv_logger = CSVDataLogger(filename=raw_filename)
                print(f"📝 Logger de dados brutos: {raw_filename}")
            
            # Configurar conversor para formato OpenBCI
            if self.save_converted_data:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                converted_filename = f"openbci_data_bci_{timestamp}.csv"
                self.udp_converter = RealtimeUDPConverter(output_filename=converted_filename)
                print(f"🔄 Conversor OpenBCI: {converted_filename}")
            
            # Configurar preditor BCI
            if self.enable_predictions:
                if not self.model_path.exists():
                    raise FileNotFoundError(f"Modelo não encontrado: {self.model_path}")
                
                self.bci_predictor = RealtimeBCIPredictor(
                    model_path=str(self.model_path),
                    prediction_callback=self._enhanced_prediction_callback
                )
                print(f"🧠 Preditor BCI configurado")
            
            print("✅ Todos os componentes configurados")
            
        except Exception as e:
            print(f"❌ Erro ao configurar componentes: {e}")
            raise
    
    def _enhanced_prediction_callback(self, prediction_result):
        """Callback aprimorado para predições"""
        self.stats['predictions_made'] += 1
        
        # Callback padrão
        default_prediction_callback(prediction_result)
        
        # Estatísticas adicionais
        if self.stats['predictions_made'] % 10 == 0:
            elapsed_time = time.time() - self.stats['start_time']
            pred_rate = self.stats['predictions_made'] / elapsed_time if elapsed_time > 0 else 0
            
            print(f"\n📊 ESTATÍSTICAS DO SISTEMA:")
            print(f"   Tempo decorrido: {elapsed_time:.1f}s")
            print(f"   Pacotes UDP: {self.stats['total_packets']}")
            print(f"   Amostras brutas salvas: {self.stats['raw_samples_saved']}")
            print(f"   Amostras convertidas: {self.stats['converted_samples_saved']}")
            print(f"   Predições realizadas: {self.stats['predictions_made']}")
            print(f"   Taxa de predições: {pred_rate:.2f} pred/s\n")
    
    def _process_udp_data(self, data):
        """Processar dados UDP em todos os componentes"""
        try:
            self.stats['total_packets'] += 1
            
            # Processar no logger de dados brutos
            if self.csv_logger:
                self.csv_logger.process_udp_data(data)
                self.stats['raw_samples_saved'] += 1
            
            # Processar no conversor OpenBCI
            if self.udp_converter:
                self.udp_converter.process_udp_data(data)
                self.stats['converted_samples_saved'] += 1
            
            # Processar no preditor BCI
            if self.bci_predictor:
                self.bci_predictor._process_udp_data(data)
                
        except Exception as e:
            print(f"❌ Erro ao processar dados UDP: {e}")
    
    def start(self):
        """Iniciar o sistema completo"""
        try:
            print(f"\n🚀 INICIANDO SISTEMA BCI COMPLETO")
            print(f"=" * 60)
            
            # Configurar componentes
            self._setup_components()
            
            # Iniciar receptor UDP
            self.udp_receiver.start(callback=self._process_udp_data)
            
            # Marcar como rodando
            self.running = True
            self.stats['start_time'] = time.time()
            
            print(f"✅ Sistema iniciado com sucesso!")
            print(f"📡 Aguardando dados UDP do OpenBCI GUI...")
            print(f"💡 Pressione Ctrl+C para parar")
            
            # Loop principal
            try:
                while self.running:
                    time.sleep(1)
                    
                    # Verificar se ainda está recebendo dados
                    if self.stats['total_packets'] == 0 and time.time() - self.stats['start_time'] > 10:
                        print("⚠️ Nenhum dado UDP recebido nos últimos 10 segundos")
                        print("💡 Verifique se o OpenBCI GUI está enviando dados para localhost:12345")
                    
            except KeyboardInterrupt:
                print(f"\n⏹️ Parando sistema...")
                self.stop()
                
        except Exception as e:
            print(f"❌ Erro ao iniciar sistema: {e}")
            self.stop()
            raise
    
    def stop(self):
        """Parar o sistema completo"""
        try:
            print(f"🛑 Parando componentes...")
            
            self.running = False
            
            # Parar receptor UDP
            if self.udp_receiver:
                self.udp_receiver.stop()
                print(f"   ✅ Receptor UDP parado")
            
            # Fechar logger de dados brutos
            if self.csv_logger:
                self.csv_logger.close()
                print(f"   ✅ Logger de dados brutos fechado")
            
            # Fechar conversor OpenBCI
            if self.udp_converter:
                self.udp_converter.close()
                print(f"   ✅ Conversor OpenBCI fechado")
            
            # Parar preditor BCI
            if self.bci_predictor:
                self.bci_predictor.stop_udp_processing()
                print(f"   ✅ Preditor BCI parado")
            
            # Mostrar estatísticas finais
            self._show_final_stats()
            
            print(f"✅ Sistema parado com sucesso!")
            
        except Exception as e:
            print(f"❌ Erro ao parar sistema: {e}")
    
    def _show_final_stats(self):
        """Mostrar estatísticas finais"""
        if self.stats['start_time']:
            elapsed_time = time.time() - self.stats['start_time']
            
            print(f"\n📊 ESTATÍSTICAS FINAIS:")
            print(f"   Tempo total: {elapsed_time:.1f}s")
            print(f"   Pacotes UDP recebidos: {self.stats['total_packets']}")
            print(f"   Amostras brutas salvas: {self.stats['raw_samples_saved']}")
            print(f"   Amostras convertidas: {self.stats['converted_samples_saved']}")
            print(f"   Predições realizadas: {self.stats['predictions_made']}")
            
            if elapsed_time > 0:
                print(f"   Taxa de pacotes: {self.stats['total_packets'] / elapsed_time:.2f} pkt/s")
                print(f"   Taxa de predições: {self.stats['predictions_made'] / elapsed_time:.2f} pred/s")

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description='Sistema BCI Completo')
    parser.add_argument('--model', type=str, required=True, help='Caminho para o modelo .pt')
    parser.add_argument('--host', type=str, default='localhost', help='Host UDP')
    parser.add_argument('--port', type=int, default=12345, help='Porta UDP')
    parser.add_argument('--no-raw', action='store_true', help='Não salvar dados brutos')
    parser.add_argument('--no-converted', action='store_true', help='Não salvar dados convertidos')
    parser.add_argument('--no-predictions', action='store_true', help='Não fazer predições')
    
    args = parser.parse_args()
    
    # Verificar se o modelo existe
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Modelo não encontrado: {model_path}")
        print(f"💡 Modelos disponíveis:")
        models_dir = Path.cwd() / "models"
        if models_dir.exists():
            for model_file in models_dir.glob("*.pt"):
                print(f"   - {model_file}")
        return
    
    # Criar sistema BCI
    bci_system = CompleteBCISystem(
        model_path=str(model_path),
        save_raw_data=not args.no_raw,
        save_converted_data=not args.no_converted,
        enable_predictions=not args.no_predictions,
        udp_host=args.host,
        udp_port=args.port
    )
    
    # Iniciar sistema
    bci_system.start()

if __name__ == "__main__":
    main()
