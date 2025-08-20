
> **Objetivo**: Gerar código confiável e fácil de manter. Sempre: planejar → implementar o menor passo viável → cobrir com testes → executar/verificar → refatorar. Seguir SOLID e manter o design minimalista.

## Estilo e Filosofia
- **Minimalismo**: prefira a solução mais simples que atende ao requisito; evite overengineering.
- **SOLID**:  
  - *S*: cada classe/módulo com uma responsabilidade clara.  
  - *O*: aberto a extensão, fechado a modificação; use interfaces/abstrações.  
  - *L*: preserve contratos ao substituir tipos (sem quebrar expectativas).  
  - *I*: interfaces pequenas e específicas; evite “interfaces gordas”.  
  - *D*: dependa de abstrações; injete dependências (sem new/instâncias rígidas espalhadas).
- **Legibilidade > esperteza**: nomes descritivos, funções curtas, evitar side effects inesperados.
- **Erros previsíveis**: valide entradas cedo; mensagens de erro claras; nunca engula exceções silenciosamente.

## Fluxo Obrigatório (quando gerar código)
1. **Planeje em 3–6 bullets** o menor incremento possível.  
2. **Proponha a API** (assinaturas, tipos, contratos) antes do corpo.  
3. **Teste primeiro** (ou junto):  
   - Se linguagem for **Python**: `pytest` com *arrange–act–assert*.  
   - Se **TypeScript/JS**: `vitest` ou `jest`.  
   - Se **Java**: JUnit 5.  
   - Cobrir casos felizes + bordas + falhas previsíveis.  
4. **Implemente** apenas o necessário para passar os testes.  
5. **Execute/verifique**: mostre o comando para rodar testes/linters (ex.: `pytest -q`, `npm test`, `mvn -q -DskipTests=false test`).  
6. **Refatore** mantendo verde; extraia funções/classes quando necessário para cumprir SOLID.  
7. **Docs curtas**: docstrings/comentários só onde agregam (contratos, invariantes, decisões).  
8. **Exemplos de uso** (snippet de 3–10 linhas) quando criar APIs públicas.

## Padrões de Qualidade
- **Cobertura de testes alvo**: funções novas com testes; se criar camada de infra, isole com *ports/adapters* e use *fakes/mocks* onde fizer sentido.  
- **Lint/Format**: respeitar ferramentas do projeto (ex.: `ruff`/`black`, `eslint`/`prettier`, `spotless`). Se ausentes, sugerir configuração mínima.  
- **Contratos**: valide precondições com asserts/guard clauses; documente invariantes.  
- **Acoplamento baixo**: injete dependências via construtor ou fábrica; evite singletons e globals.  
- **Coesão alta**: classes e módulos pequenos, focados.

## Arquitetura (padronizar camadas)
- **Domínio**: entidades, value objects, serviços de domínio (puro, sem dependência de infra).  
- **Aplicação/Use Cases**: orquestra domínio; portas (interfaces) para persistência/IO.  
- **Infra/Adapters**: implementa portas (DB, HTTP, FS, mensagens).  
- **Interface** (CLI/HTTP/UI): validação/serialização/DTOs → chama casos de uso.

> O Copilot deve **manter dependências fluindo para o domínio (inbound)** e **detalhes para as bordas (outbound)**.

## Instruções Específicas por Linguagem (curtas)
- **Python**: tipagem com `typing`/`pydantic` (se já no projeto), `pytest`, `ruff` + `black`.  
- **TypeScript**: `strict` true; types primeiro; `vitest`/`jest`; `eslint` + `prettier`.  
- **Java**: JDK 17+; `record` para DTOs; JUnit 5; *dependency injection* leve (ex.: construtor).  
*(Se o repositório já define outras ferramentas, seguir as existentes.)*

## Como quero que você responda (Copilot)
- **Sempre**: comece com um **plano numerado curto**.  
- Entregue **primeiro os testes**, depois a implementação mínima.  
- Inclua **comandos para executar testes/linters**.  
- Ao finalizar, liste um **checklist** com o que foi feito e o que fica de pendência.  
- Se algo estiver ambíguo, **faça hipóteses explícitas** e siga com a solução mínima.

## Checklists para PR/Commit
- [ ] Testes novos/verdes; casos de borda cobertos.  
- [ ] Nomeação clara; funções/métodos <= ~30 linhas.  
- [ ] Nenhum acoplamento desnecessário; dependências injetadas.  
- [ ] Sem *dead code*; logs e erros significativos.  
- [ ] Documentação mínima (contratos/decisões).
