# Estrutura da Pasta SRC

## Arquivos Principais (Pasta `src/`)

### `bci_interface.py` 
**Arquivo principal do sistema**
- Interface PyQt5 para o Sistema BCI
- Funcionalidades principais:
  - Cadastro de pacientes
  - Streaming de dados em tempo real com visualização
  - Gravação de dados atrelada ao paciente com marcadores T1, T2 e Baseline
- **Dependências**: Todos os outros arquivos listados abaixo

### `config.py`
**Configurações do sistema**
- Funções para gerenciamento de caminhos
- Configuração de pastas de dados
- Usado por: `bci_interface.py`

### `openbci_csv_logger.py`
**Logger especializado para dados OpenBCI**
- Gravação de dados EEG em formato CSV
- Suporte a marcadores temporais
- Usado por: `bci_interface.py`

### `udp_receiver.py`
**Receptor de dados UDP**
- Recebe dados em tempo real via UDP
- Interface para comunicação com hardware
- Usado por: `bci_interface.py`

### `realtime_udp_converter.py`
**Conversor de dados UDP**
- Processa dados UDP recebidos
- Converte para formato EEG utilizável
- Usado por: `bci_interface.py`

### `csv_data_logger.py`
**Logger CSV genérico**
- Alternativa de fallback para gravação
- Logger simples para dados EEG
- Usado por: `bci_interface.py`

## Arquivos Arquivados (Pasta `src/extra/`)

### `bci_minimal.py`
- Versão simplificada da interface BCI
- **Status**: Não utilizado pelo sistema principal

### `dual_capture_system.py`
- Sistema de captura dupla
- **Status**: Não utilizado pelo sistema principal

### `realtime_bci_system_v2.py`
- Versão anterior do sistema BCI
- **Status**: Substituído por `bci_interface.py`

### `bci_interface_old.py`
- Versão anterior da interface principal
- **Status**: Backup da versão anterior

## Como Executar

Para executar o sistema principal:

```bash
cd src
python bci_interface.py
```

## Dependências Externas

O sistema requer as seguintes bibliotecas Python:
- PyQt5
- numpy
- pandas
- matplotlib
- sqlite3 (built-in)

## Estrutura de Dependências

```
bci_interface.py (MAIN)
├── config.py
├── openbci_csv_logger.py
├── udp_receiver.py
├── realtime_udp_converter.py
└── csv_data_logger.py
```

Todos os arquivos na pasta `extra/` são independentes e não são necessários para o funcionamento do sistema principal.
