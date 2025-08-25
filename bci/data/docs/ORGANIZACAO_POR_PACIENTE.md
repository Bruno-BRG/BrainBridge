# Sistema BCI - Organização por Paciente

## 🆕 Nova Funcionalidade: Pastas Individuais por Paciente

Cada paciente agora tem sua própria pasta dentro de `data/recordings/`, facilitando a organização e o gerenciamento dos dados.

### 📁 Estrutura de Pastas

```
data/recordings/
├── P001_João_Silva/
│   ├── P001_baseline_20250708_143022.csv
│   ├── P001_treino_20250708_143500.csv
│   ├── P001_teste_20250708_144000.csv
│   └── P001_jogo_20250708_144500.csv
├── P002_Maria_Santos/
│   ├── P002_baseline_20250708_145000.csv
│   └── P002_treino_20250708_145300.csv
└── P003_Pedro_Oliveira/
    └── P003_teste_20250708_150000.csv
```

### 🔧 Como Funciona

1. **Criação Automática**: Quando você inicia uma gravação, o sistema:
   - Cria automaticamente a pasta do paciente (se não existir)
   - Sanitiza o nome do paciente (remove caracteres especiais)
   - Salva o arquivo na pasta correta

2. **Seleção de Tarefa**: 
   - **Interface**: Dropdown com opções padronizadas: "Baseline", "Treino", "Teste", "Jogo"
   - **Conversão**: Automaticamente convertido para formato de arquivo (ex: "Treino" → "treino")
   - **Vantagem**: Evita erros de digitação e padroniza nomenclatura

3. **Nomenclatura**:
   - **Pasta**: `P{ID}_Nome_Sanitizado` (ex: `P001_João_Silva`)
   - **Arquivo**: `P{ID}_{tarefa}_{timestamp}.csv` (ex: `P001_treino_20250708_143022.csv`)

4. **Interface**: 
   - Mostra o caminho relativo: `P001_João_Silva/P001_treino_20250708_143022.csv`
   - Facilita identificação visual do paciente e tarefa

### ✅ Vantagens

- **Organização Clara**: Cada paciente tem seus dados separados
- **Backup Fácil**: Copie a pasta de um paciente específico
- **Histórico Completo**: Todas as sessões de um paciente juntas
- **Busca Rápida**: Encontre facilmente dados de um paciente
- **Manutenção**: Easier para limpar ou arquivar dados antigos

### 🔄 Compatibilidade

- **Formato OpenBCI**: 100% mantido
- **Arquivos Existentes**: Continuam funcionando (ficam na raiz)
- **Novos Arquivos**: Automaticamente organizados em pastas
- **Interface**: Funciona transparentemente

### 💡 Exemplo de Uso

1. **Cadastrar Paciente**: João Silva (ID será P001)
2. **Selecionar Tarefa**: Escolher "Treino" no dropdown
3. **Iniciar Gravação**: Sistema cria pasta e arquivo automaticamente
4. **Resultado**: Arquivo salvo em `data/recordings/P001_João_Silva/P001_treino_TIMESTAMP.csv`
5. **Próxima Sessão**: Escolher "Teste" → arquivo na mesma pasta `P001_João_Silva/`

### 🎯 Tarefas Disponíveis

- **Baseline**: Para medições de referência em repouso
- **Treino**: Para sessões de treinamento e aprendizado
- **Teste**: Para avaliações e testes de desempenho  
- **Jogo**: Para sessões com interfaces de jogos/aplicações

### 🧪 Testado e Validado

- ✅ Criação automática de pastas
- ✅ Sanitização de nomes com caracteres especiais
- ✅ Múltiplas sessões por paciente
- ✅ Formato OpenBCI preservado
- ✅ Interface funcionando sem erros
- ✅ Compatibilidade com sistema existente

**🎉 O sistema agora está mais organizado e profissional!**
