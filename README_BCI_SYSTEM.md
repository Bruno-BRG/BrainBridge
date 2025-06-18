# Sistema BCI - Brain-Computer Interface

Sistema completo para interface cérebro-computador com gestão de pacientes, aquisição de dados LSL e inferência em tempo real.

## Características

### 🏥 Gestão de Pacientes
- Cadastro completo de pacientes com informações médicas
- Histórico de sessões
- Acompanhamento de gravações
- Banco de dados SQLite integrado

### 📡 Aquisição de Dados LSL
- Conexão com streams LSL (Lab Streaming Layer)
- Gravação em tempo real com formato OpenBCI
- Marcação de movimentos (mão esquerda/direita) com 400 amostras
- Salvamento automático em CSV

### 🧠 Inferência em Tempo Real
- Carregamento de modelos EEGInceptionERP pré-treinados
- Predição em tempo real de movimentos de mão
- Visualização de confiança e probabilidades
- Buffer deslizante para análise contínua

## Instalação

### 1. Instalar Dependências

```bash
pip install -r requirements_gui.txt
```

### 2. Executar o Sistema

```bash
python run_bci_system.py
```

## Estrutura do Projeto

```
projetoBCI/
├── src/
│   ├── database.py           # Gerenciador de banco de dados
│   ├── lsl_streamer.py      # Módulo de streaming LSL
│   ├── inference_engine.py  # Motor de inferência
│   └── main_gui.py          # Interface principal PyQt
├── models/
│   └── teste/               # Modelos pré-treinados
├── requirements_gui.txt     # Dependências
└── run_bci_system.py       # Executável principal
```

## Como Usar

### 1. Gestão de Pacientes

1. **Adicionar Paciente**: Clique em "Adicionar Paciente" e preencha os dados
2. **Editar Paciente**: Selecione na lista e edite os campos
3. **Visualizar Sessões**: Sessões aparecem automaticamente ao selecionar paciente

### 2. Aquisição LSL

1. **Conectar Stream**: Clique "Conectar ao Stream" para encontrar streams LSL
2. **Selecionar Paciente**: Escolha o paciente na lista
3. **Iniciar Gravação**: Clique "Iniciar Gravação"
4. **Marcar Movimentos**: 
   - "Mão Esquerda" para movimento da mão esquerda (400 amostras)
   - "Mão Direita" para movimento da mão direita (400 amostras)
5. **Parar Gravação**: Clique "Parar Gravação" para salvar

### 3. Inferência em Tempo Real

1. **Conectar LSL**: Clique "Conectar LSL"
2. **Carregar Modelo**: Selecione modelo e clique "Carregar Modelo"
3. **Iniciar Inferência**: Clique "Iniciar Inferência"
4. **Visualizar Resultados**: Acompanhe predições em tempo real

## Formato dos Dados

### Arquivo CSV de Gravação

O sistema grava dados no formato OpenBCI:

```csv
%OpenBCI Raw EXG Data
%Number of channels = 16
%Sample Rate = 125 Hz
%Board = LSL_Stream

Sample Index,EXG Channel 0,EXG Channel 1,...,EXG Channel 15,Timestamp,Annotations
1,-0.5,10.2,9.8,...,-1.3,0.008,T0
2,-8.5,30.3,16.2,...,10.4,0.016,
3,8.5,44.5,31.2,...,30.6,0.024,T1
...
```

### Anotações

- **T0**: Estado de repouso
- **T1**: Movimento da mão esquerda
- **T2**: Movimento da mão direita

## Modelos Suportados

- **EEGInceptionERP**: Modelo principal (via braindecode)
- **FallbackCNN**: Modelo alternativo caso braindecode não esteja disponível

### Parâmetros do Modelo

- **Canais**: 16 (EEG)
- **Taxa de Amostragem**: 125 Hz
- **Janela Temporal**: 4 segundos (500 amostras)
- **Classes**: 2 (Mão Esquerda vs Mão Direita)

## Banco de Dados

O sistema usa SQLite com as seguintes tabelas:

- **patients**: Informações dos pacientes
- **sessions**: Sessões de gravação
- **recordings**: Arquivos gravados
- **models**: Modelos treinados

## Pré-processamento em Tempo Real

1. **Filtro Passa-Banda**: 8-30 Hz
2. **Normalização Z-score**: Por canal
3. **Janela Deslizante**: 500 amostras (4 segundos)

## Configuração LSL

Para usar com OpenBCI:

1. Instale OpenBCI GUI
2. Configure stream LSL no OpenBCI GUI
3. Configure tipo "EEG" e 16 canais
4. Inicie o stream antes de conectar no sistema

## Troubleshooting

### Erro "Stream não encontrado"
- Verifique se o stream LSL está ativo
- Confirme se o tipo é "EEG"
- Teste com `pylsl.resolve_streams()` no Python

### Erro de modelo
- Verifique se o arquivo .pt existe em `models/teste/`
- Confirme compatibilidade PyTorch
- Teste carregamento manual do modelo

### Erro de banco de dados
- Verifique permissões de escrita no diretório
- Delete `bci_system.db` para recriar

## Desenvolvimento

### Adicionar Novo Modelo

1. Coloque arquivo .pt em `models/teste/`
2. Modifique `inference_engine.py` se necessário
3. Ajuste parâmetros no banco de dados

### Extensões

- Adicione novos tipos de anotação em `lsl_streamer.py`
- Implemente novos filtros em `inference_engine.py`
- Adicione campos ao banco em `database.py`

## Contribuição

1. Fork o projeto
2. Crie branch para feature
3. Commit mudanças
4. Push para branch
5. Abra Pull Request

## Licença

Este projeto é para fins acadêmicos e de pesquisa.
