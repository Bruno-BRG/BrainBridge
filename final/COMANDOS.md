# Comandos Rápidos - Sistema BCI

## Testar Modelo
```bash
python final/test_model_loading.py
```

## Sistema Completo com Dados Reais (OpenBCI GUI)
```bash
python final/complete_bci_system.py --model models/custom_eegnet_1751389051.pt
```

## Apenas Predições (sem salvar dados)
```bash
python final/complete_bci_system.py --model models/custom_eegnet_1751389051.pt --no-raw --no-converted
```

## Apenas Captura de Dados (sem predições)
```bash
python final/complete_bci_system.py --model models/custom_eegnet_1751389051.pt --no-predictions
```

## Demonstração com Dados Simulados
```bash
python final/demo_complete_system.py --model models/custom_eegnet_1751389051.pt --duration 30
```

## Apenas Simulador EEG
```bash
python final/eeg_simulator.py --duration 30
```

## Sistema de Predição Standalone
```bash
python final/realtime_bci_system.py --model models/custom_eegnet_1751389051.pt
```

## Captura Dual (Raw + OpenBCI)
```bash
python final/dual_capture_system.py
```

## Testar com Dados Simulados (Terminal 1)
```bash
python final/eeg_simulator.py --duration 60
```

## Testar com Dados Simulados (Terminal 2)
```bash
python final/realtime_bci_system.py --model models/custom_eegnet_1751389051.pt
```

## Configurações Avançadas

### Mudar Porta UDP
```bash
python final/complete_bci_system.py --model models/custom_eegnet_1751389051.pt --port 12346
```

### Configurar Host Remoto
```bash
python final/complete_bci_system.py --model models/custom_eegnet_1751389051.pt --host 192.168.1.100
```

### Simulador com Configurações Específicas
```bash
python final/eeg_simulator.py --sample-rate 250 --channels 8 --duration 120
```

## Troubleshooting

### Erro de Porta em Uso
```bash
netstat -ano | findstr :12345
```

### Verificar Modelos Disponíveis
```bash
dir models\*.pt
```

### Logs Detalhados
Adicionar logging ao script:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Fluxo de Trabalho Recomendado

1. **Testar modelo**:
   ```bash
   python final/test_model_loading.py
   ```

2. **Demo com dados simulados**:
   ```bash
   python final/demo_complete_system.py --model models/custom_eegnet_1751389051.pt --duration 30
   ```

3. **Usar com OpenBCI GUI**:
   - Configure OpenBCI GUI para enviar UDP para localhost:12345
   - Execute:
   ```bash
   python final/complete_bci_system.py --model models/custom_eegnet_1751389051.pt
   ```

4. **Análise dos dados**:
   - Arquivos CSV gerados na pasta final/
   - Formato: `raw_data_bci_YYYYMMDD_HHMMSS.csv`
   - Formato: `openbci_data_bci_YYYYMMDD_HHMMSS.csv`
