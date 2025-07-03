"""
Sistema de captura UDP -> CSV em tempo real
Para uso com OpenBCI GUI ou outros sistemas de streaming UDP
"""

from csv_data_logger import CSVDataLogger
import time

def main():
    """
    Sistema de captura UDP -> CSV para dados do OpenBCI GUI
    """
    print("🚀 Sistema de Captura UDP -> CSV em Tempo Real")
    print("=" * 60)
    print("📡 Compatível com OpenBCI GUI e outros sistemas UDP")
    print("=" * 60)
    
    # Criar logger com configurações otimizadas para tempo real
    logger = CSVDataLogger()
    
    # Configurações para tempo real com OpenBCI
    logger.set_buffer_size(25)      # Salvar a cada 25 registros (bom para EEG)
    logger.set_auto_flush_interval(2)  # Ou a cada 2 segundos (dados críticos)
    
    try:
        # Iniciar captura
        logger.start_logging()
        
        print(f"📡 Sistema aguardando dados UDP em localhost:12345")
        print(f"💾 Arquivo CSV: {logger.csv_filename}")
        print(f"⚡ Salvamento: a cada 25 registros OU a cada 2 segundos")
        print()
        print("📋 INSTRUÇÕES:")
        print("   1. Abra o OpenBCI GUI")
        print("   2. Configure networking para enviar UDP para localhost:12345")
        print("   3. Inicie o streaming no OpenBCI GUI")
        print("   4. Os dados aparecerão aqui automaticamente!")
        print()
        print("🛑 Pressione Ctrl+C para parar")
        print("=" * 60)
        
        # Loop de monitoramento em tempo real
        counter = 0
        last_count = 0
        
        while True:
            time.sleep(1)
            counter += 1
            
            # Mostrar status a cada 3 segundos
            if counter % 3 == 0:
                stats = logger.get_stats()
                new_messages = stats['total_received'] - last_count
                last_count = stats['total_received']
                
                status_icon = "🟢" if stats['total_received'] > 0 else "🔴"
                rate_icon = "⚡" if new_messages > 0 else "⏸️"
                
                print(f"[{counter:03d}s] {status_icon} Total: {stats['total_received']:4d} | "
                      f"Buffer: {stats['buffer_size']:2d} | "
                      f"Arquivo: {stats['file_size_bytes']:6d} bytes | "
                      f"{rate_icon} Taxa: ~{new_messages*20}/min")
                
                # Dicas se não estiver recebendo dados
                if counter == 30 and stats['total_received'] == 0:
                    print("\n� DICA: Se não estiver recebendo dados:")
                    print("   • Verifique se o OpenBCI GUI está rodando")
                    print("   • Confirme se o UDP está configurado para localhost:12345")
                    print("   • Certifique-se que o streaming foi iniciado\n")
                
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("🛑 Parando sistema...")
        logger.stop_logging()
        
        # Estatísticas finais
        stats = logger.get_stats()
        print("📋 ESTATÍSTICAS FINAIS:")
        print(f"   • Total capturado: {stats['total_received']} mensagens")
        print(f"   • Arquivo salvo: {stats['csv_file']}")
        print(f"   • Tamanho final: {stats['file_size_bytes']} bytes")
        print(f"   • Último save: {stats['last_save']}")
        print("✅ Sistema parado com sucesso!")
        print("\n💡 Seu arquivo CSV está pronto para análise!")


if __name__ == "__main__":
    main()
