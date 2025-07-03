"""
Sistema Dual de Captura UDP em Tempo Real
Executa simultaneamente:
1. CSV Data Logger (dados brutos UDP)
2. OpenBCI Converter (formato padrão OpenBCI)
"""

from csv_data_logger import CSVDataLogger
from realtime_udp_converter import RealTimeUDPConverter
from udp_receiver import UDPReceiver
import time
from datetime import datetime

def main():
    """
    Sistema dual de captura e conversão em tempo real
    """
    print("🚀 SISTEMA DUAL DE CAPTURA UDP EM TEMPO REAL")
    print("=" * 70)
    print("📡 Captura dados UDP e gera 2 arquivos simultaneamente:")
    print("   1️⃣ CSV Bruto (dados UDP originais)")
    print("   2️⃣ CSV OpenBCI (formato padrão para análise)")
    print("=" * 70)
    
    # Configurar timestamp único para ambos os arquivos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Criar UM receptor UDP compartilhado
    shared_udp_receiver = UDPReceiver()
    
    # Criar ambos os sistemas SEM receptor próprio
    csv_logger = CSVDataLogger(csv_filename=f"udp_raw_{timestamp}.csv")
    openbci_converter = RealTimeUDPConverter(csv_filename=f"openbci_converted_{timestamp}.csv")
    
    # Substituir os receptores internos pelo compartilhado
    csv_logger.udp_receiver = shared_udp_receiver
    openbci_converter.udp_receiver = shared_udp_receiver
    
    # Configurar parâmetros otimizados
    csv_logger.set_buffer_size(30)
    csv_logger.set_auto_flush_interval(3)
    
    # Função para processar dados para ambos os sistemas
    def dual_callback(data):
        """Callback que envia dados para ambos os sistemas"""
        try:
            csv_logger._process_udp_data(data)
            openbci_converter._process_udp_data(data)
        except Exception as e:
            print(f"Erro no callback dual: {e}")
    
    try:
        # Iniciar ambos os sistemas SEM iniciar seus receptores UDP individuais
        print("🔧 Iniciando sistemas...")
        
        # Configurar flags de logging/converting antes de iniciar
        csv_logger.is_logging = True
        openbci_converter.is_converting = True
        
        # Configurar o callback dual no receptor compartilhado
        shared_udp_receiver.set_callback(dual_callback)
        
        # Iniciar APENAS o receptor compartilhado
        shared_udp_receiver.start()
        
        print(f"📡 Aguardando dados UDP em localhost:12345")
        print(f"📁 Arquivo UDP Bruto: {csv_logger.csv_filename}")
        print(f"📁 Arquivo OpenBCI: {openbci_converter.csv_filename}")
        print()
        print("📋 INSTRUÇÕES:")
        print("   1. Configure OpenBCI GUI para enviar UDP para localhost:12345")
        print("   2. Inicie o streaming no OpenBCI GUI")
        print("   3. Ambos os arquivos serão gerados automaticamente!")
        print()
        print("🛑 Pressione Ctrl+C para parar")
        print("=" * 70)
        
        # Loop de monitoramento
        counter = 0
        last_udp_count = 0
        last_samples_count = 0
        
        while True:
            time.sleep(2)
            counter += 2
            
            # Obter estatísticas de ambos os sistemas
            csv_stats = csv_logger.get_stats()
            openbci_stats = openbci_converter.get_stats()
            
            # Usar stats do receptor compartilhado para contagem UDP
            udp_total = shared_udp_receiver.get_data_count()
            
            # Calcular taxas
            new_udp = udp_total - last_udp_count
            new_samples = openbci_stats['total_samples_converted'] - last_samples_count
            last_udp_count = udp_total
            last_samples_count = openbci_stats['total_samples_converted']
            
            # Status visual
            status_udp = "🟢" if udp_total > 0 else "🔴"
            status_convert = "🟢" if openbci_stats['total_samples_converted'] > 0 else "🔴"
            
            print(f"[{counter:03d}s] {status_udp} UDP: {udp_total:4d} (+{new_udp:2d}) | "
                  f"{status_convert} Samples: {openbci_stats['total_samples_converted']:5d} (+{new_samples:3d}) | "
                  f"Erros: {openbci_stats['conversion_errors']:2d}")
            
            # Mostrar detalhes a cada 10 segundos
            if counter % 10 == 0:
                print("─" * 70)
                print(f"📊 ESTATÍSTICAS ({counter}s):")
                print(f"   📥 UDP Bruto    : {csv_stats['buffer_size']:2d} buffer | {csv_stats['file_size_bytes']:7d} bytes")
                print(f"   🔄 OpenBCI     : {openbci_stats['buffer_size']:2d} buffer | {openbci_stats['file_size_bytes']:7d} bytes")
                print(f"   ⚡ Taxa atual  : {new_udp*30}/min UDP, {new_samples*30}/min samples")
                print("─" * 70)
                
            # Dicas se não estiver recebendo dados
            if counter == 30 and udp_total == 0:
                print("\n💡 DICA: Se não estiver recebendo dados:")
                print("   • Verifique se o OpenBCI GUI está rodando")
                print("   • Confirme se o UDP está configurado para localhost:12345")
                print("   • Certifique-se que o streaming foi iniciado")
                print("   • Teste com: telnet localhost 12345\n")
                
    except KeyboardInterrupt:
        print("\n" + "=" * 70)
        print("🛑 Parando sistemas...")
        
        # Parar o receptor compartilhado
        shared_udp_receiver.stop()
        
        # Parar processamento dos sistemas individuais
        csv_logger.is_logging = False
        openbci_converter.is_converting = False
        
        # Forçar salvamento final
        if csv_logger.data_buffer:
            csv_logger.force_save()
        if openbci_converter.sample_buffer:
            openbci_converter.force_save()
        
        # Estatísticas finais
        csv_stats = csv_logger.get_stats()
        openbci_stats = openbci_converter.get_stats()
        udp_total = shared_udp_receiver.get_data_count()
        
        print("📋 ESTATÍSTICAS FINAIS:")
        print("─" * 70)
        print("📥 SISTEMA UDP BRUTO:")
        print(f"   • Total UDP recebidos: {udp_total}")
        print(f"   • Arquivo: {csv_stats['csv_file']}")
        print(f"   • Tamanho: {csv_stats['file_size_bytes']} bytes")
        print()
        print("🔄 SISTEMA OPENBCI CONVERTER:")
        print(f"   • Total samples convertidos: {openbci_stats['total_samples_converted']}")
        print(f"   • Arquivo: {openbci_stats['csv_file']}")
        print(f"   • Tamanho: {openbci_stats['file_size_bytes']} bytes")
        print(f"   • Erros de conversão: {openbci_stats['conversion_errors']}")
        print("─" * 70)
        print("✅ Ambos os sistemas parados com sucesso!")
        print()
        print("📁 ARQUIVOS GERADOS:")
        print(f"   1️⃣ {csv_stats['csv_file']}")
        print(f"   2️⃣ {openbci_stats['csv_file']}")
        print()
        print("💡 Use o arquivo OpenBCI para análise em softwares compatíveis!")


if __name__ == "__main__":
    main()
