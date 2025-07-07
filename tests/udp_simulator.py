"""
Simulador UDP para teste da interface BCI
Envia dados EEG simulados via UDP
"""
import socket
import struct
import time
import numpy as np
import threading
from datetime import datetime

class UDPSimulator:
    """Simulador de dados EEG via UDP"""
    
    def __init__(self, host='127.0.0.1', port=12345, sample_rate=125):
        self.host = host
        self.port = port
        self.sample_rate = sample_rate
        self.running = False
        self.socket = None
        
    def start(self):
        """Inicia o simulador"""
        self.running = True
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        print(f"Iniciando simulador UDP em {self.host}:{self.port}")
        print(f"Taxa de amostragem: {self.sample_rate} Hz")
        
        # Iniciar thread de envio
        self.thread = threading.Thread(target=self._send_loop)
        self.thread.start()
        
    def stop(self):
        """Para o simulador"""
        self.running = False
        if self.socket:
            self.socket.close()
            
    def _send_loop(self):
        """Loop principal de envio de dados"""
        sample_interval = 1.0 / self.sample_rate  # Intervalo entre amostras
        sample_count = 0
        
        print("Enviando dados EEG simulados...")
        
        while self.running:
            try:
                # Gerar dados EEG simulados para 16 canais
                eeg_data = self._generate_eeg_sample(sample_count)
                
                # Empacotar dados no formato UDP
                packet = self._pack_samples([eeg_data])
                
                # Enviar para interface
                self.socket.sendto(packet, (self.host, self.port))
                
                sample_count += 1
                
                # Log a cada 125 amostras (1 segundo)
                if sample_count % 125 == 0:
                    print(f"Enviadas {sample_count} amostras ({sample_count/125:.1f}s)")
                
                # Aguardar próxima amostra
                time.sleep(sample_interval)
                
            except Exception as e:
                if self.running:
                    print(f"Erro no envio: {e}")
                break
                
    def _generate_eeg_sample(self, sample_count):
        """Gera uma amostra EEG simulada realística"""
        # Parâmetros para simulação
        t = sample_count / self.sample_rate
        
        eeg_sample = []
        
        for channel in range(16):
            # Componentes de frequência típicas do EEG
            alpha = 10 * np.sin(2 * np.pi * 10 * t)  # Ritmo alfa (10 Hz)
            beta = 5 * np.sin(2 * np.pi * 20 * t)    # Ritmo beta (20 Hz)
            gamma = 2 * np.sin(2 * np.pi * 40 * t)   # Ritmo gama (40 Hz)
            
            # Ruído e drift
            noise = np.random.randn() * 5
            drift = 0.1 * np.sin(2 * np.pi * 0.1 * t)
            
            # Variação por canal
            channel_factor = 1 + 0.2 * channel
            
            # Sinal final
            signal = (alpha + beta + gamma + noise + drift) * channel_factor
            eeg_sample.append(signal)
            
        return eeg_sample
    
    def _pack_samples(self, samples):
        """Empacota amostras no formato UDP esperado"""
        # Formato: número de amostras (uint32) + amostras (16 floats cada)
        num_samples = len(samples)
        packet = struct.pack('I', num_samples)
        
        for sample in samples:
            packet += struct.pack('16f', *sample)
            
        return packet

def main():
    """Função principal"""
    print("=== Simulador UDP para BCI ===")
    print("Este simulador envia dados EEG fictícios para testar a interface")
    print("Pressione Ctrl+C para parar")
    
    simulator = UDPSimulator()
    
    try:
        simulator.start()
        
        # Manter o programa rodando
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nParando simulador...")
        simulator.stop()
        print("Simulador parado!")

if __name__ == "__main__":
    main()
