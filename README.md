# Sistema BCI - Interface PyQt5 com Marcadores

## 🚀 Como Executar (ATUALIZADO)

### ✨ Opção 1: Script Principal (Recomendado)
```bash
cd projetoBCI
python run_bci.py
```

### ✨ Opção 2: Como Módulo Python
```bash
cd projetoBCI
python -m bci
```

### ✨ Opção 3: Usando o módulo bci diretamente
```python
import bci
bci.run()
```

## 📁 Estrutura Reorganizada

```
projetoBCI/
├── 🎯 run_bci.py                    # ← SCRIPT PRINCIPAL
├── bci/                             # Pacote principal organizado
│   ├── __init__.py                  # Exports do pacote
│   ├── main.py                      # Ponto de entrada limpo
│   ├── BCI_main_window.py           # Janela principal
│   ├── streaming_widget.py          # Interface de streaming
│   ├── patient_registration_widget.py # Cadastro de pacientes
│   └── ... (outros módulos)
├── src/                             # Diretório legacy (depreciado)
├── data/, models/, docs/            # Dados e documentação
└── README.md                        # Este arquivo
```

### ✅ Interface Completa
- **Cadastro de Pacientes**: Nome, idade, sexo, mão afetada, tempo desde evento
- **Streaming em Tempo Real**: Visualização de dados EEG com gráficos sliding window
- **Gravação Vinculada**: Arquivos CSV ligados ao paciente selecionado
- **Marcadores T1/T2**: Botões para inserir marcadores durante a gravação
- **Baseline Automático**: Timer de 5 minutos que bloqueia outros marcadores
- **Auto T0**: Inserção automática de T0 após 400 amostras dos marcadores T1/T2

### 🗂️ Estrutura Organizada
```
projeto/
├── src/                    # Código fonte principal
│   ├── bci_interface.py   # Interface principal PyQt5
│   ├── config.py          # Configuração de caminhos
│   ├── udp_receiver.py    # Receptor UDP
│   └── ...                # Outros módulos
├── data/                  # Dados e gravações
│   ├── recordings/        # Arquivos CSV de gravação
│   └── database/          # Banco SQLite de pacientes
├── tests/                 # Scripts de teste
├── docs/                  # Documentação
├── legacy/                # Arquivos antigos
└── models/                # Modelos ML
```

## 🚀 Como Usar

### 1. Instalação das Dependências
```bash
pip install PyQt5 numpy pandas matplotlib sqlite3
```

### 2. Executar a Interface
```bash
cd src
python bci_interface.py
```

### 3. Teste da Interface
```bash
cd tests
python test_interface_markers.py
```

## 🎮 Uso da Interface

### Aba "Cadastro de Pacientes"
1. Preencher dados do paciente
2. Clicar em "Cadastrar Paciente"
3. Visualizar lista de pacientes cadastrados

### Aba "Streaming e Gravação"

#### Conexão
1. Configurar Host (localhost) e Porta (12345)
2. Clicar "Conectar" (modo simulação se UDP não disponível)

#### Gravação
1. Selecionar paciente na lista
2. Clicar "Iniciar Gravação"
3. Os botões de marcadores são habilitados

#### Marcadores
- **T1** (Azul): Movimento Real
  - Insere marcador T1 imediatamente
  - Programa T0 para 400 amostras depois
  
- **T2** (Laranja): Movimento Imaginado
  - Insere marcador T2 imediatamente
  - Programa T0 para 400 amostras depois
  
- **Baseline** (Roxo): Período de Repouso
  - Insere marcador BASELINE
  - Bloqueia outros botões por 5 minutos
  - Mostra countdown timer

## 📊 Formato dos Arquivos CSV

```csv
Timestamp,EXG Channel 0,EXG Channel 1,...,EXG Channel 15,Marker
2024-01-01T10:00:00.000,1.23,4.56,...,7.89,
2024-01-01T10:00:00.004,2.34,5.67,...,8.90,T1
2024-01-01T10:00:01.600,3.45,6.78,...,9.01,T0
2024-01-01T10:02:00.000,4.56,7.89,...,0.12,BASELINE
```

## 🔧 Funcionalidades Técnicas

### SimpleCSVLogger
- Gravação thread-safe
- Buffer para otimização
- Suporte a marcadores
- Auto-inserção de T0

### Timer de Baseline
- 5 minutos (300 segundos)
- Countdown visual
- Bloqueio automático de botões
- Reabilitação automática

### Estrutura de Banco
```sql
-- Tabela de pacientes
patients: id, name, age, sex, affected_hand, time_since_event, created_at, notes

-- Tabela de gravações
recordings: id, patient_id, filename, start_time, notes
```

## 🧪 Modo de Simulação

Se o UDP não estiver disponível, o sistema automaticamente entra em modo simulação:
- Dados EEG simulados (ruído gaussiano)
- Status "Simulação (Dados Fake)"
- Todas as funcionalidades funcionam normalmente

## 📝 Arquivos de Saída

### Gravações
- Local: `data/recordings/`
- Formato: `patient_{id}_{nome}_{timestamp}.csv`
- Conteúdo: 16 canais EEG + timestamp + marcadores

### Banco de Dados
- Local: `data/database/bci_patients.db`
- Tabelas: patients, recordings
- Backup automático recomendado

## ⚡ Próximos Passos

1. ✅ Interface PyQt5 completa
2. ✅ Marcadores T1, T2, Baseline
3. ✅ Timer de 5 minutos
4. ✅ Auto T0 após 400 amostras
5. ✅ Estrutura de pastas organizada
6. 🔄 Integração com modelos ML
7. 🔄 Análise em tempo real
8. 🔄 Exportação para formatos padrão (EDF, etc.)

## 🐛 Solução de Problemas

### Erro de Import
```bash
# Se módulos não forem encontrados
cd src
python bci_interface.py
```

### Porta UDP Ocupada
- O sistema detecta automaticamente
- Entra em modo simulação
- Funcionalidades completas mantidas

### Dependências
```bash
pip install --upgrade PyQt5 numpy pandas matplotlib
```

## 📞 Suporte

Para problemas ou sugestões:
1. Verificar logs no terminal
2. Testar modo simulação
3. Verificar estrutura de pastas
4. Reinstalar dependências se necessário

---

**Status**: ✅ Funcional - Interface completa com marcadores implementados
**Última atualização**: Janeiro 2024
