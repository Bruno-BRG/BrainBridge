# Sistema BCI - Implementação Completa

## 📋 Resumo Final

O sistema BCI foi completamente implementado com todas as funcionalidades solicitadas:

### ✅ Funcionalidades Implementadas

#### 1. **Estrutura de Projeto Organizada**
- Diretório `src/` para código fonte
- Diretório `data/recordings/` para gravações (organizadas por paciente)
- Diretório `data/db/` para banco de dados
- Diretório `tests/` para testes
- Diretório `docs/` para documentação

#### 2. **Interface de Streaming/Gravação com PyQt5**
- Interface com 2 abas principais:
  - **Cadastro de Pacientes**: Registro completo de pacientes
  - **Streaming e Gravação**: Captura e gravação de dados EEG

#### 3. **Registro de Pacientes**
- Cadastro completo com: nome, idade, sexo, mão afetada, tempo desde evento
- Banco de dados SQLite para persistência
- Lista visual de pacientes cadastrados

#### 4. **Visualização EEG em Tempo Real**
- Plot em tempo real de dados EEG (16 canais)
- Visualização por canal individual ou todos os canais
- Escalas ajustáveis (Auto, ±50µV, ±100µV, ±200µV, ±500µV)
- Janela deslizante de 8 segundos

#### 5. **Sistema de Marcadores (T1, T2, T0)**
- **T1**: Movimento Real (azul)
- **T2**: Movimento Imaginado (laranja)
- **T0**: Inserido automaticamente 400 amostras após T1/T2
- Feedback visual quando marcadores são adicionados

#### 6. **Funcionalidade de Baseline**
- Botão específico para iniciar baseline (roxo)
- **Bloqueia T1 e T2 por 5 minutos** quando ativado
- Timer visual mostrando tempo restante do baseline
- Reabilita marcadores automaticamente após 5 minutos

#### 7. **Formato CSV OpenBCI Compatível**
- Header exato do OpenBCI
- Ordem de colunas: Sample Index, EXG Channels 0-15, Accel X/Y/Z, Other, Analog, Timestamp, Annotations
- Sistema de marcadores na coluna "Annotations"
- Compatibilidade total com dados do OpenBCI

#### 8. **Organização por Paciente**
- Cada paciente tem sua própria pasta em `data/recordings/`
- Formato de pastas: `P001_NomePaciente/`
- Arquivo CSV por sessão: `P001_baseline_20250108_143022.csv`

#### 9. **Campo de Tarefa (Dropdown)**
- Dropdown com opções fixas:
  - **Baseline**
  - **Treino** 
  - **Teste**
  - **Jogo**
- Valor usado no nome do arquivo (em lowercase)

#### 10. **⏱️ Timer de Sessão**
- **Inicia automaticamente** ao começar gravação
- **Para automaticamente** ao terminar gravação
- Display em formato **HH:MM:SS**
- Atualização em tempo real (a cada segundo)
- Posicionado ao lado do status de gravação

#### 11. **🎮 Modo Jogo (NOVO)**
- **Botão inteligente**: Muda para "Iniciar Jogo" quando tarefa "Jogo" está selecionada
- **Estados dinâmicos**:
  - Jogo não ativo: **"Iniciar Jogo"**
  - Jogo ativo: **"Parar Jogo"**
  - Outras tarefas: **"Iniciar Gravação"** / **"Parar Gravação"**
- **Interface contextual**: Labels e mensagens se adaptam ao modo jogo

### 🔧 Arquivos Principais

```
src/
├── bci_interface.py          # Interface principal PyQt5
├── openbci_csv_logger.py     # Logger compatível com OpenBCI
├── config.py                 # Configuração de caminhos
├── udp_receiver.py           # Receptor UDP para dados
├── realtime_udp_converter.py # Conversor de dados UDP
└── csv_data_logger.py        # Logger CSV simples

data/
├── recordings/               # Gravações organizadas por paciente
│   ├── P001_João/           # Pasta do paciente
│   │   ├── P001_baseline_20250108_143022.csv
│   │   └── P001_treino_20250108_144015.csv
│   └── P002_Maria/          # Outro paciente
└── db/
    └── patients.db          # Banco de dados SQLite

tests/
├── test_openbci_logger.py   # Teste do logger OpenBCI
├── test_patient_folders.py  # Teste organização por paciente
├── test_task_dropdown.py    # Teste dropdown de tarefas
├── test_timer_simple.py     # Teste lógica do timer
└── test_complete_validation.py # Validação completa
```

### 🎯 Como Usar

1. **Executar o sistema**:
   ```bash
   cd src
   python bci_interface.py
   ```

2. **Cadastrar paciente**:
   - Aba "Cadastro de Pacientes"
   - Preencher dados e clicar "Cadastrar"

3. **Iniciar gravação**:
   - Aba "Streaming e Gravação"
   - Conectar ao UDP (ou usar simulação)
   - Selecionar paciente e tarefa
   - **Para modo Jogo**: Botão muda para "Iniciar Jogo"
   - **Para outras tarefas**: Botão permanece "Iniciar Gravação"
   - **Timer de sessão inicia automaticamente**

4. **Usar marcadores**:
   - **T1**: Para movimento real
   - **T2**: Para movimento imaginado
   - **Baseline**: Bloqueia outros marcadores por 5 min
   - **T0**: Inserido automaticamente após T1/T2

5. **Parar gravação**:
   - **Para modo Jogo**: Clicar "Parar Jogo"
   - **Para outras tarefas**: Clicar "Parar Gravação"
   - **Timer de sessão para automaticamente**
   - Arquivo salvo na pasta do paciente

### 📊 Formato do Arquivo CSV

```csv
Sample Index,EXG Channel 0,EXG Channel 1,...,EXG Channel 15,Accel X,Accel Y,Accel Z,Other,Analog,Timestamp,Annotations
0,-0.123,0.456,...,1.789,0.0,0.0,0.0,0,0,1735934400.123,
1,-0.234,0.567,...,1.890,0.0,0.0,0.0,0,0,1735934400.131,
...
400,-0.345,0.678,...,1.901,0.0,0.0,0.0,0,0,1735934403.323,T1
...
800,-0.456,0.789,...,2.012,0.0,0.0,0.0,0,0,1735934406.523,T0
```

### ✅ Testes Implementados

Todos os testes passam:
- ✅ Logger OpenBCI compatível
- ✅ Organização por pastas de paciente
- ✅ Dropdown de tarefas funcional
- ✅ Timer de sessão implementado
- ✅ Modo jogo com botão inteligente
- ✅ Interface completa funcionando
- ✅ Sistema de marcadores correto

### 🚀 Sistema Pronto para Uso!

O sistema BCI está **completamente implementado** e **testado**, incluindo:
- ✅ Timer de sessão com start/stop automático
- ✅ Todas as funcionalidades originais
- ✅ Compatibilidade total com OpenBCI
- ✅ Interface intuitiva e robusta

**Última atualização**: Modo Jogo implementado com botão inteligente! 🎮⏱️
