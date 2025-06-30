# 🧠 Sistema BCI - Scripts de Teste em Tempo Real

Este conjunto de scripts permite testar seu modelo BCI treinado com dados de stream LSL em tempo real.

## 📁 Arquivos Criados

### 1. `validate_model.py` - Validação Básica
```bash
python validate_model.py
```
**O que faz:**
- ✅ Encontra automaticamente o modelo mais recente em `models/`
- ✅ Carrega o modelo e verifica se está funcionando
- ✅ Testa com dados sintéticos realistas  
- ✅ Mostra se o modelo está pronto para uso com LSL
- ✅ **Execute este primeiro** para verificar se tudo está OK

### 2. `simple_bci_test.py` - Teste LSL Simples
```bash
python simple_bci_test.py
```
**O que faz:**
- 🔍 Procura automaticamente streams LSL na rede
- 📊 Coleta buffers de **400 frames** (3.2s @ 125Hz)
- 🧠 Faz predições: **0 = Mão Esquerda**, **1 = Mão Direita**
- 📈 Mostra estatísticas em tempo real
- ⏱️ Roda por 60 segundos (Ctrl+C para parar antes)

### 3. `test_lsl_realtime_bci.py` - Sistema Completo
```bash
python test_lsl_realtime_bci.py
```
**Sistema BCI completo com:**
- 🎮 Simulador LSL integrado (`--simulate`)
- 📋 Listagem de streams (`--list-streams`)
- ⚙️ Configurações avançadas
- 📊 Estatísticas detalhadas
- 🔧 Suporte a múltiplos tipos de modelo

## 🚀 Como Usar

### Passo 1: Validar Modelo
```bash
# Verificar se o modelo está funcionando
python validate_model.py
```

### Passo 2: Configurar Stream LSL
Você precisa de uma fonte de dados EEG via LSL:

**Opção A: OpenBCI GUI**
1. Abra OpenBCI GUI
2. Configure 16 canais @ 125Hz
3. Ative "Networking" → "LSL Stream" 
4. Start streaming

**Opção B: Simulador (para testes)**
```bash
# Terminal 1: Rodar simulador
python test_lsl_realtime_bci.py --simulate

# Terminal 2: Rodar teste
python simple_bci_test.py
```

**Opção C: Outros dispositivos**
- BrainVision actiCHamp
- g.tec g.Nautilus  
- Emotiv EPOC
- Qualquer dispositivo compatível com LSL

### Passo 3: Executar Teste
```bash
# Teste simples (recomendado)
python simple_bci_test.py

# Ou sistema completo
python test_lsl_realtime_bci.py
```

## 📊 Saída Esperada

```
🧠 #1: Mão Esquerda (confiança: 0.847)
📊 Buffer: 400/400 (100.0%)
🧠 #2: Mão Direita (confiança: 0.923)
📊 Buffer: 400/400 (100.0%)
🧠 #3: Mão Esquerda (confiança: 0.756)
...

📊 RESULTADOS FINAIS
===================
⏱️ Tempo total: 60.0s
🔢 Predições: 28
👈 Mão Esquerda: 15 (53.6%)
👉 Mão Direita: 13 (46.4%)
```

## ⚙️ Configurações

### Parâmetros do Buffer
- **400 frames** = 3.2 segundos @ 125Hz
- **16 canais** EEG
- Buffer circular para coleta contínua

### Normalização
- Usa estatísticas salvas do treinamento (se disponível)
- Fallback: normalização z-score em tempo real

### Performance
- **~10-50ms** por predição (depende do hardware)
- **CPU/GPU** automático
- **Memória mínima** para operação contínua

## 🔧 Solução de Problemas

### "Nenhum stream LSL encontrado"
```bash
# Verificar streams disponíveis
python test_lsl_realtime_bci.py --list-streams

# Testar com simulador
python test_lsl_realtime_bci.py --simulate
```

### "Modelo não encontrado"
```bash
# Verificar pasta models/
ls models/

# Treinar modelo (execute o notebook primeiro)
```

### "Canais incompatíveis"
O sistema adapta automaticamente:
- Se stream tem mais canais → usa primeiros 16
- Se stream tem menos canais → preenche com zeros

### "Taxa de amostragem diferente"
O sistema detecta automaticamente, mas pode afetar performance.
Recomendado: 125Hz

## 📝 Logs

Todos os scripts salvam logs em:
- Console (tempo real)
- `bci_test.log` (arquivo)

## 🎯 Interpretação dos Resultados

### Confiança
- **> 0.8**: Alta confiança
- **0.6-0.8**: Confiança média  
- **< 0.6**: Baixa confiança

### Distribuição
- **50/50**: Ideal para dados balanceados
- **Muito desbalanceado**: Possível problema no modelo ou dados

### Variação
- **Sempre mesma classe**: Modelo não está discriminando
- **Variação realista**: Modelo funcionando

## 📚 Arquitetura do Sistema

```
Stream LSL → Buffer (400 frames) → Normalização → Modelo → Predição (0/1)
    ↓              ↓                    ↓           ↓          ↓
  OpenBCI      Circular           Z-score     EEGNet    Confiança
   125Hz       Queue              Stats      PyTorch    Softmax
```

## 🔄 Integração com Sistema Principal

Estes scripts são compatíveis com:
- `src/inference_engine.py`
- `src/lsl_streamer.py` 
- `src/main_gui.py`

Para integrar ao sistema principal, use as classes e funções destes scripts como referência.

---

**💡 Dica**: Execute sempre `validate_model.py` primeiro para garantir que tudo está funcionando antes de usar com dados reais!
