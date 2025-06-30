# 🧠 Sistema BCI - Instruções Completas de Teste

## 📋 Resumo dos Scripts Criados

Criei 4 scripts completos para testar seu sistema BCI em tempo real:

### 1. `validate_model.py` ⭐ **EXECUTE PRIMEIRO**
Valida se o modelo está funcionando corretamente:
```bash
python validate_model.py
```

### 2. `simple_bci_test.py` 🎯 **TESTE BÁSICO**
Teste simples com LSL (60 segundos):
```bash
python simple_bci_test.py
```

### 3. `final_bci_test.py` 🚀 **TESTE COMPLETO**
Sistema BCI completo com estatísticas avançadas:
```bash
python final_bci_test.py
```

### 4. `test_lsl_realtime_bci.py` 🔧 **SISTEMA AVANÇADO**
Sistema completo com simulador e opções avançadas:
```bash
python test_lsl_realtime_bci.py
```

## 🎯 O Que Cada Script Faz

### Sistema de Buffer
- **400 frames** por predição = **3.2 segundos** @ 125Hz
- **Buffer circular** para coleta contínua
- **16 canais** EEG (adapta automaticamente se diferente)

### Normalização
- Usa **estatísticas exatas do treinamento** (se disponível)
- Fallback para **z-score** em tempo real
- **Garante compatibilidade** com o modelo treinado

### Predições
- **Saída**: `0` = Mão Esquerda, `1` = Mão Direita
- **Confiança**: probabilidade da predição (0-1)
- **Tempo de processamento**: latência em milissegundos

## 🚀 Como Testar (Passo a Passo)

### OPÇÃO A: Com Hardware EEG Real

1. **Configure seu dispositivo EEG** (OpenBCI, etc.):
   - 16 canais @ 125Hz
   - Ative streaming LSL
   - Verifique se está enviando dados

2. **Valide o modelo**:
   ```bash
   python validate_model.py
   ```

3. **Execute o teste**:
   ```bash
   python final_bci_test.py
   ```

### OPÇÃO B: Sem Hardware (Simulação)

1. **Terminal 1 - Simulador**:
   ```bash
   python test_lsl_realtime_bci.py --simulate
   ```

2. **Terminal 2 - Teste**:
   ```bash
   python final_bci_test.py
   ```

## 📊 Interpretação dos Resultados

### Exemplo de Saída Esperada:
```
🧠 # 1 [  2.0s]: Mão Esquerda  (conf: 0.847, proc: 12.3ms)
🧠 # 2 [  4.0s]: Mão Direita   (conf: 0.923, proc: 11.8ms)
🧠 # 3 [  6.0s]: Mão Esquerda  (conf: 0.756, proc: 13.1ms)
📊 Últimas 10: 6L/4R | Conf: 0.834 | 12.4ms

📊 RESULTADOS FINAIS
⏱️ Duração: 60.0s
🔢 Predições: 30
👈 Mão Esquerda: 18 (60.0%)
👉 Mão Direita: 12 (40.0%)
🎯 Confiança: 0.834
⚡ Processamento: 12.4ms
✅ Modelo produz predições variadas
```

### Indicadores de Sucesso:

✅ **Bom funcionamento**:
- Predições variadas (não sempre a mesma classe)
- Confiança > 0.6
- Tempo de processamento < 50ms
- Distribuição não muito desbalanceada (30-70% é OK)

⚠️ **Possíveis problemas**:
- Sempre a mesma classe → modelo enviesado ou dados uniformes
- Confiança muito baixa (< 0.5) → modelo incerto
- Tempo alto (> 100ms) → possível problema de performance

## 🔧 Solução de Problemas

### "Nenhum modelo encontrado"
```bash
ls models/  # Verificar se há arquivos .pt
```
→ Execute o notebook de treinamento primeiro

### "Nenhum stream LSL encontrado"
```bash
python test_lsl_realtime_bci.py --list-streams
```
→ Use o simulador ou configure dispositivo EEG

### "Sempre a mesma predição"
Isso pode ser normal se:
- Você não está fazendo movimento/imaginando
- Dados são de repouso (sem comando motor)
- Modelo foi treinado em dados específicos

Para testar com variação:
- Use o simulador que gera padrões diferentes
- Experimente imaginar movimentos durante o teste

### "Erro de normalização"
Os scripts têm fallbacks automáticos, mas se persistir:
- Use `simple_bci_test.py` que usa normalização mais robusta

## 📈 Performance Esperada

### Latência Total (buffer → predição):
- **Coleta**: 3.2s (buffer de 400 frames)
- **Processamento**: 10-50ms
- **Total**: ~3.25s por predição

### Acurácia:
- **Modelo treinado**: Depende dos dados de treinamento
- **Dados reais**: Pode variar conforme qualidade do sinal
- **Simulação**: Aleatório (50/50 é esperado)

## 🎯 Próximos Passos

Após validar que tudo funciona:

1. **Integração com GUI**: Use as classes destes scripts no `main_gui.py`

2. **Controle de Dispositivos**: Adicione ações baseadas nas predições:
   ```python
   if prediction == 0:
       # Ação para mão esquerda
       control_left_device()
   else:
       # Ação para mão direita  
       control_right_device()
   ```

3. **Coleta de Dados**: Use para coletar novos dados para retreinamento

4. **Monitoramento**: Adicione logging e métricas para uso prolongado

## 📝 Logs e Debug

Todos os scripts salvam logs em:
- **Console**: Saída em tempo real
- **bci_test.log**: Arquivo de log (alguns scripts)

Para debug avançado, edite o nível de logging:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 🎉 Conclusão

Você agora tem um **sistema BCI completo** que:
- ✅ Carrega seu modelo treinado automaticamente
- ✅ Conecta a streams LSL em tempo real
- ✅ Processa buffers de 400 frames como especificado
- ✅ Faz predições 0/1 com confiança
- ✅ Mostra estatísticas detalhadas
- ✅ Funciona com dados reais ou simulados

**Execute `python final_bci_test.py` para começar!** 🚀
