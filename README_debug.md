# 🔍 Debug Stream LSL → Modelo BCI

Scripts de teste e debug para verificar a compatibilidade entre o stream LSL e o modelo BCI treinado.

## 📁 Arquivos Criados

### 1. `debug_stream_to_model.py` 
**Script principal de debug visual**

- 🎯 **Objetivo**: Testar e visualizar todo o pipeline Stream LSL → Modelo
- 📊 **Funcionalidades**:
  - Recebe stream LSL igual ao script de reforço
  - Segue protocolo T0→T1→T0→T2→T0
  - Debug visual detalhado em tempo real
  - Verificação de shapes em cada etapa
  - Análise estatística dos dados
  - Plots em tempo real (6 subplots)

### 2. `test_model_import.py`
**Teste de importação do modelo**

- 🎯 **Objetivo**: Verificar se o modelo pode ser importado corretamente
- 🧪 **Testes**:
  - Importação do notebook
  - Criação de modelo local (fallback)
  - Forward pass com diferentes shapes
  - Carregamento de modelos salvos

## 🚀 Como Usar

### Passo 1: Testar Importação do Modelo
```bash
python test_model_import.py
```
Este script vai:
- ✅ Verificar se consegue importar `CustomEEGModel` do notebook
- ✅ Testar criação e forward pass
- ✅ Verificar modelos salvos

### Passo 2: Debug Visual do Stream
```bash
python debug_stream_to_model.py
```
Este script vai:
- 🔍 Carregar o modelo mais recente
- 📡 Conectar ao stream LSL
- 🔄 Executar 2 ciclos (10 predições)
- 📊 Mostrar visualizações em tempo real
- 🧠 Debug completo de cada etapa

## 📊 Visualizações em Tempo Real

O `debug_stream_to_model.py` mostra 6 plots em tempo real:

1. **📊 EEG Raw**: Sinais brutos (primeiros 8 canais)
2. **📈 EEG Normalizado**: Sinais após normalização
3. **🎯 Distribuição por Canal**: Média e desvio por canal
4. **⚡ Tensor Shape Flow**: Transformações de formato
5. **🧠 Saída do Modelo**: Probabilidades das classes
6. **📊 Estatísticas**: Resumo geral

## 🔍 Debug Detalhado

### Logs Incluem:
- **📥 Coleta**: Taxa de amostragem, tempo de coleta
- **🔄 Normalização**: Shapes, ranges, estatísticas
- **⚡ Tensor**: Conversões de formato (3D/4D)
- **🧠 Modelo**: Forward pass, probabilidades
- **📊 Resultado**: Predição final e confiança

### Exemplo de Log:
```
🔍 PROCESSAMENTO COMPLETO DEBUG:
============================================================
1️⃣ DADOS RAW:
   - Shape: (16, 400)
   - Range: [-0.000152, 0.000184]
   - Mean: 0.000003

2️⃣ NORMALIZAÇÃO:
   - Input range: [-0.000152, 0.000184]
   - Output range: [-2.456, 3.122]
   - ✅ Normalização salva aplicada

3️⃣ CONVERSÃO PARA TENSOR:
   - Normalized shape: (16, 400)
   - Tensor 3D: (1, 16, 400)
   - Tensor 4D: (1, 1, 16, 400)

4️⃣ FORWARD PASS:
   - ✅ Formato 3D funcionou!
   - Output shape: (1, 2)
   - Predição: Mão Direita
   - Confiança: 0.7234
```

## 🎯 Protocolo de Teste

### Sequência T (igual ao reinforcement):
- **T0**: Repouso (não usado para reforço)
- **T1**: Mão Esquerda (label 0)
- **T2**: Mão Direita (label 1)
- **Ciclo**: T0→T1→T0→T2→T0 (5 predições)

### Coleta de Dados:
- **📥 Tamanho**: 400 amostras sequenciais
- **🔄 Sem sobreposição**: Próxima coleta continua de onde parou
- **⏱️ Taxa**: 125Hz (esperado)

## 🛠️ Solução de Problemas

### Erro de Importação do Modelo:
```python
# Se falhar importação do notebook, usar definição local
class CustomEEGModel(nn.Module):
    # Definição de fallback incluída
```

### Problemas de Shape:
- ✅ O script testa automaticamente formatos 3D e 4D
- ✅ Mostra exatamente qual formato funciona
- ✅ Debug completo de todas as transformações

### Stream LSL não encontrado:
- 🔍 Script procura por 'EEG', 'ExG', 'EMG'
- 📡 Mostra informações detalhadas do stream
- ⚠️ Logs de troubleshooting incluídos

## 📋 Checklist de Verificação

Após executar os scripts, verifique:

- [ ] ✅ Modelo pode ser importado/criado
- [ ] ✅ Forward pass funciona
- [ ] ✅ Stream LSL conecta
- [ ] ✅ Dados são coletados na taxa correta
- [ ] ✅ Normalização funciona
- [ ] ✅ Shapes são compatíveis
- [ ] ✅ Predições são geradas
- [ ] ✅ Visualizações aparecem

## 🎯 Próximos Passos

Se tudo funcionar:
1. ✅ Execute `reinforcement_bci_test.py` 
2. ✅ O pipeline está validado
3. ✅ Dados chegam corretamente ao modelo

Se houver problemas:
1. 🔍 Analise os logs detalhados
2. 📊 Verifique as visualizações
3. 🛠️ Ajuste baseado no debug

## 📝 Logs Salvos

- **debug_stream.log**: Log completo do debug
- **Console**: Output em tempo real com emojis

---

**🎯 Objetivo**: Garantir 100% de compatibilidade entre Stream LSL e Modelo BCI antes do treinamento por reforço!
