# 🧠 Sistema BCI - Guia de Execução

## ✅ Formas Corretas de Executar

### 1. **Script Principal (Mais Simples)**
```bash
cd projetoBCI
python run_bci.py
```

### 2. **Como Módulo Python**
```bash
cd projetoBCI
python -m bci
```

### 3. **Executar main.py diretamente**
```bash
cd projetoBCI
python bci/main.py
```

### 4. **Via função do módulo**
```python
import bci
bci.run()
```

## ❌ Problemas Comuns e Soluções

### Erro: "attempted relative import with no known parent package"

**Causa**: Tentar executar arquivos com imports relativos diretamente

**❌ Não faça isso:**
```bash
python c:/caminho/completo/para/bci/main.py
```

**✅ Faça assim:**
```bash
cd projetoBCI
python bci/main.py
# ou
python run_bci.py
```

### Erro: "No module named 'bci'"

**Causa**: Executando de diretório errado

**✅ Solução:**
```bash
cd projetoBCI  # ← Importante estar na raiz do projeto
python run_bci.py
```

### Erro: "ImportError: No module named 'PyQt5'"

**✅ Solução:**
```bash
pip install -r requirements.txt
# ou
pip install PyQt5 torch numpy matplotlib
```

## 🎯 Recomendação

**Use sempre o script principal:**
```bash
cd projetoBCI
python run_bci.py
```

Este é o método mais confiável e fornece melhor feedback de erro.

## 🔧 Para Desenvolvimento

Se você estiver desenvolvendo e quiser testar apenas importações:

```python
# Testar imports
python -c "import bci; print('OK')"

# Testar classes específicas  
python -c "from bci import BCIMainWindow; print('OK')"

# Testar função run
python -c "import bci; print('run disponível:', hasattr(bci, 'run'))"
```

## 📁 Estrutura de Diretórios

Certifique-se de que está na estrutura correta:
```
📁 projetoBCI/          ← Você deve estar AQUI
├── 🐍 run_bci.py       ← Execute este arquivo
├── 📁 bci/
│   ├── main.py
│   ├── BCI_main_window.py
│   └── ...
├── 📁 data/
├── 📁 models/
└── requirements.txt
```
