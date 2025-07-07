# Configuração do Sistema BCI

## Estrutura do Projeto

- `src/` - Código fonte principal
  - `bci_interface.py` - Interface principal PyQt5
  - `udp_receiver.py` - Receptor UDP para dados EEG
  - `realtime_udp_converter.py` - Conversor de dados em tempo real
  - `csv_data_logger.py` - Logger para arquivos CSV

- `data/` - Dados e gravações
  - `recordings/` - Gravações de pacientes (arquivos CSV)
  - `database/` - Banco de dados SQLite dos pacientes

- `docs/` - Documentação do projeto

- `tests/` - Scripts de teste e demonstração

- `legacy/` - Arquivos antigos do projeto

- `models/` - Modelos de ML treinados

## Uso

Para executar a interface principal:
```bash
cd src
python bci_interface.py
```

Para executar testes:
```bash
cd tests
python test_interface.py
```
