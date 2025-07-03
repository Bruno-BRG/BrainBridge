"""
Simulador de Dados EEG para Teste do Sistema BCI
Gera dados simulados no formato OpenBCI e envia via UDP para testar o sistema
"""

import json
import time
import socket
import numpy as np
import threading
from datetime import datetime
from typing import Optional
import argparse

class EEGDataSimulator:
    """Simulador de dados EEG no formato OpenBCI"""
    
    def __init__(self, 
                 host: str = 'localhost',
                 port: int = 12345,
                 sample_rate: float = 125.0,
                 n_channels: int = 16):
        
        self.host = host
        self.port = port
        self.sample_rate = sample_rate
        self.n_channels = n_channels
        self.running = False
        
        # Configuração do socket UDP
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Parâmetros de simulação
        self.time_step = 1.0 / sample_rate  # Intervalo entre amostras
        self.sample_count = 0
        
        # Simulação de padrões EEG
        self.base_frequency = 10.0  # Alfa (10 Hz)
        self.noise_level = 0.1
        
        # Padrões para diferentes classes
        self.left_hand_pattern = self._generate_pattern_weights('left')
        self.right_hand_pattern = self._generate_pattern_weights('right')
        
        # Estado atual da simulação
        self.current_pattern = None
        self.pattern_duration = 0
        self.pattern_start_time = 0
        
        print(f"🎮 EEG Simulator iniciado: {host}:{port}")
        print(f"📊 {n_channels} canais, {sample_rate} Hz")
    
    def _generate_pattern_weights(self, pattern_type: str) -> np.ndarray:
        """Gerar pesos para simular padrões EEG"""
        weights = np.ones(self.n_channels)
        
        if pattern_type == 'left':
            # Simular maior atividade nos canais do lado direito do cérebro (C3, C4)
            # Canais 7-10 representam área motora
            weights[7:10] *= 2.0  # Maior atividade
            weights[2:5] *= 0.5   # Menor atividade contralateral
        elif pattern_type == 'right':
            # Simular maior atividade nos canais do lado esquerdo do cérebro
            weights[2:5] *= 2.0   # Maior atividade
            weights[7:10] *= 0.5  # Menor atividade contralateral
        
        return weights
    
    def _generate_eeg_sample(self) -> np.ndarray:
        """Gerar uma amostra EEG sintética"""
        # Tempo atual
        t = self.sample_count * self.time_step
        
        # Sinal base (ritmo alfa)
        alpha_signal = np.sin(2 * np.pi * self.base_frequency * t)
        
        # Adicionar harmônicos
        beta_signal = 0.5 * np.sin(2 * np.pi * 20 * t)  # Beta (20 Hz)
        gamma_signal = 0.2 * np.sin(2 * np.pi * 40 * t)  # Gamma (40 Hz)
        
        # Combinar sinais
        base_signal = alpha_signal + beta_signal + gamma_signal
        
        # Gerar amostra para todos os canais
        sample = np.ones(self.n_channels) * base_signal
        
        # Aplicar padrão específico se ativo
        if self.current_pattern is not None:
            if self.current_pattern == 'left':
                sample *= self.left_hand_pattern
            elif self.current_pattern == 'right':
                sample *= self.right_hand_pattern
        
        # Adicionar ruído
        noise = np.random.normal(0, self.noise_level, self.n_channels)
        sample += noise
        
        # Escalar para valores típicos de EEG (microvolts)
        sample *= 50.0  # Amplitude típica de EEG
        
        # Adicionar deriva lenta
        drift = 10 * np.sin(2 * np.pi * 0.1 * t)  # Deriva de 0.1 Hz
        sample += drift
        
        return sample
    
    def _update_pattern(self):
        """Atualizar padrão atual baseado no tempo"""
        current_time = time.time()
        
        # Verificar se precisa mudar de padrão
        if (self.current_pattern is None or 
            current_time - self.pattern_start_time >= self.pattern_duration):
            
            # Escolher novo padrão aleatoriamente
            patterns = [None, 'left', 'right']  # None = repouso
            weights = [0.5, 0.25, 0.25]  # Mais tempo em repouso
            
            self.current_pattern = np.random.choice(patterns, p=weights)
            self.pattern_start_time = current_time
            
            # Duração do padrão (2-5 segundos)
            self.pattern_duration = np.random.uniform(2.0, 5.0)
            
            pattern_name = {
                None: "Repouso",
                'left': "Mão Esquerda",
                'right': "Mão Direita"
            }
            
            print(f"🧠 Padrão: {pattern_name[self.current_pattern]} por {self.pattern_duration:.1f}s")
    
    def _create_udp_packet(self, eeg_samples: list) -> str:
        """Criar pacote UDP no formato OpenBCI"""
        packet = {
            "timeSeriesRaw": eeg_samples,
            "timestamp": datetime.now().isoformat(),
            "sampleIndex": self.sample_count,
            "boardType": "OpenBCI_GUI_Simulator",
            "sampleRate": self.sample_rate,
            "channels": self.n_channels
        }
        
        return json.dumps(packet)
    
    def start_simulation(self, duration: Optional[float] = None):
        """Iniciar simulação"""
        self.running = True
        start_time = time.time()
        
        print(f"🚀 Simulação iniciada")
        if duration:
            print(f"⏱️ Duração: {duration} segundos")
        print(f"📡 Enviando dados para {self.host}:{self.port}")
        print(f"💡 Pressione Ctrl+C para parar")
        
        try:
            while self.running:
                # Atualizar padrão
                self._update_pattern()
                
                # Gerar amostra EEG
                eeg_sample = self._generate_eeg_sample()
                
                # Criar pacote UDP
                packet_data = self._create_udp_packet([eeg_sample.tolist()])
                
                # Enviar via UDP
                try:
                    self.sock.sendto(packet_data.encode(), (self.host, self.port))
                except Exception as e:
                    print(f"❌ Erro ao enviar UDP: {e}")
                
                # Incrementar contador
                self.sample_count += 1
                
                # Mostrar progresso
                if self.sample_count % 250 == 0:  # A cada 2 segundos
                    elapsed = time.time() - start_time
                    rate = self.sample_count / elapsed
                    print(f"📊 Amostras enviadas: {self.sample_count} (taxa: {rate:.1f} Hz)")
                
                # Verificar duração
                if duration and (time.time() - start_time) >= duration:
                    print(f"⏰ Duração atingida: {duration}s")
                    break
                
                # Aguardar próxima amostra
                time.sleep(self.time_step)
                
        except KeyboardInterrupt:
            print(f"\n⏹️ Simulação interrompida pelo usuário")
        finally:
            self.stop_simulation()
    
    def stop_simulation(self):
        """Parar simulação"""
        self.running = False
        self.sock.close()
        
        print(f"🛑 Simulação parada")
        print(f"📊 Total de amostras enviadas: {self.sample_count}")
    
    def start_background_simulation(self, duration: Optional[float] = None):
        """Iniciar simulação em background"""
        self.simulation_thread = threading.Thread(
            target=self.start_simulation,
            args=(duration,)
        )
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        return self.simulation_thread

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description='Simulador de Dados EEG')
    parser.add_argument('--host', type=str, default='localhost', help='Host UDP')
    parser.add_argument('--port', type=int, default=12345, help='Porta UDP')
    parser.add_argument('--sample-rate', type=float, default=125.0, help='Taxa de amostragem')
    parser.add_argument('--channels', type=int, default=16, help='Número de canais')
    parser.add_argument('--duration', type=float, help='Duração da simulação (segundos)')
    
    args = parser.parse_args()
    
    print(f"🎮 SIMULADOR DE DADOS EEG")
    print(f"=" * 50)
    
    # Criar simulador
    simulator = EEGDataSimulator(
        host=args.host,
        port=args.port,
        sample_rate=args.sample_rate,
        n_channels=args.channels
    )
    
    # Iniciar simulação
    simulator.start_simulation(duration=args.duration)

if __name__ == "__main__":
    main()
