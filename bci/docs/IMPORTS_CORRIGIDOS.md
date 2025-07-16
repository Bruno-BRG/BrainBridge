# Estrutura de Imports Corrigida - Sistema BCI

## Resumo das Correções Realizadas

### 1. **Arquivo `streaming_widget.py`**
**Antes:**
```python
from .database import DatabaseManager
from .streaming import StreamingThread
from .eeg_net import EEGNet
from .plot_widget import EEGPlotWidget
from .logger import OpenBCICSVLogger, SimpleCSVLogger
from .utils import get_recording_path
```

**Depois:**
```python
from .database_manager import DatabaseManager
from .streaming_thread import StreamingThread
from .EGGNet import EEGNet
from .EEG_plot_widget import EEGPlotWidget
from .openbci_csv_logger import OpenBCICSVLogger
from .simple_csv_logger import SimpleCSVLogger
from .config import get_recording_path
```

### 2. **Arquivo `BCI_main_window.py`**
**Antes:**
```python
from database_manager import DatabaseManager
from patient_registration_widget import PatientRegistrationWidget
from streaming_widget import StreamingWidget
```

**Depois:**
```python
from .database_manager import DatabaseManager
from .patient_registration_widget import PatientRegistrationWidget
from .streaming_widget import StreamingWidget
```

### 3. **Arquivo `main.py`**
**Antes:**
```python
from config import get_recording_path, get_database_path, ensure_folders_exist
from openbci_csv_logger import OpenBCICSVLogger
```

**Depois:**
```python
from .config import get_recording_path, get_database_path, ensure_folders_exist
from .openbci_csv_logger import OpenBCICSVLogger
```

### 4. **Outros arquivos corrigidos:**
- `realtime_udp_converter.py`: `from udp_receiver import UDPReceiver` → `from .udp_receiver import UDPReceiver`
- `csv_data_logger.py`: `from udp_receiver import UDPReceiver` → `from .udp_receiver import UDPReceiver`

### 5. **Adicionado `__init__.py`**
Criado arquivo `bci/__init__.py` para tornar o diretório um pacote Python apropriado com exports organizados.

### 6. **Script de execução**
Criado `run_bci.py` como ponto de entrada principal do sistema.

## Estrutura do Projeto

```
projetoBCI/
├── bci/                          # Pacote principal
│   ├── __init__.py              # Exports do pacote
│   ├── BCI_main_window.py       # Janela principal
│   ├── config.py                # Configurações
│   ├── database_manager.py      # Gerenciador BD
│   ├── EEG_plot_widget.py       # Widget de plots
│   ├── EGGNet.py                # Modelo neural
│   ├── streaming_thread.py      # Thread de streaming
│   ├── streaming_widget.py      # Widget de streaming
│   ├── patient_registration_widget.py # Cadastro pacientes
│   ├── openbci_csv_logger.py    # Logger OpenBCI
│   ├── simple_csv_logger.py     # Logger simples
│   ├── udp_receiver.py          # Receptor UDP
│   └── ...
├── run_bci.py                   # Script principal
├── requirements.txt             # Dependências
└── README.md                    # Documentação
```

## Como Executar

### Opção 1: Script principal
```bash
cd projetoBCI
python run_bci.py
```

### Opção 2: Módulo direto
```bash
cd projetoBCI
python -m bci.main
```

### Opção 3: Importação em Python
```python
from bci import BCIMainWindow
from PyQt5.QtWidgets import QApplication
import sys

app = QApplication(sys.argv)
window = BCIMainWindow()
window.show()
sys.exit(app.exec_())
```

## Benefícios das Correções

1. **Imports Relativos Corretos**: Uso de `.` para imports dentro do pacote
2. **Nomes de Arquivos Corretos**: Correspondência exata com os arquivos reais
3. **Estrutura de Pacote**: `__init__.py` apropriado com exports organizados
4. **Ponto de Entrada Claro**: Script `run_bci.py` como entrada principal
5. **Compatibilidade**: Funciona tanto como módulo quanto como script standalone

## Verificação

Para verificar se tudo está funcionando:

```bash
cd projetoBCI
python -c "import bci; print('✓ Imports corretos!')"
python -c "from bci import BCIMainWindow, DatabaseManager; print('✓ Classes principais OK!')"
```
