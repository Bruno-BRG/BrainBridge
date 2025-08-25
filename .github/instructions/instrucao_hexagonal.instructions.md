# Instrução de Desenvolvimento — Arquitetura Hexagonal

> **Objetivo**: Entregar código confiável, testável e fácil de evoluir. Sempre: planejar → implementar o menor passo viável → cobrir com testes → executar/verificar → refatorar. Seguir SOLID, manter design minimalista e **proteger o domínio de detalhes de infraestrutura**.

## Estilo e Filosofia
- **Minimalismo**: implementar o que o caso de uso exige hoje. Sem “magia” nem generalizações prematuras.
- **Regra de Dependência (Hexagonal)**: domínio não conhece frameworks nem detalhes externos. Fluxo de dependência **para dentro** (domínio) via **portas**.
- **SOLID aplicado**  
  - *S*: cada caso de uso faz uma coisa.  
  - *O/D*: estender adicionando **adapters**; caso de uso depende de **ports** (interfaces).  
  - *L/I*: contratos claros; **ports** pequenas e específicas.  
- **Legibilidade > esperteza**: nomes descritivos, funções curtas, invariantes documentadas.
- **Erros previsíveis**: validar entradas na borda; lançar erros de domínio significativos; nunca engolir exceções.

## Fluxo Obrigatório (quando gerar código)
1. **Plano (3–6 bullets)** do menor incremento.  
2. **Propor a API**: contratos das **ports** (assinaturas, tipos, exceções) e DTOs.  
3. **Teste primeiro** (ou junto):  
   - **Domínio**: teste puro de entidades/serviços (sem mocks).  
   - **Aplicação**: testes dos **use cases** com **mocks/fakes** das **ports**.  
   - **Adapters**: testes de integração finos (ex.: repositório ↔ DB).  
4. **Implementar** só o necessário para ficar verde.
5. **Executar/verificar**: comandos de testes/linters (ex.: `pytest -q`, `npm test`, `mvn -q test`).  
6. **Refatorar em verde**: extrair módulos, alinhar nomes às **ports/use cases**.  
7. **Docs curtas**: contratos de **ports**, invariantes de domínio, decisões.
8. **Exemplos de uso**: snippet chamando o **use case** via **port** de entrada.

## Padrões de Qualidade
- **Cobertura**: toda **port** e **use case** com testes; adapters cobertos por integração.  
- **Contratos**: **ports** com precondições/poscondições explícitas; DTOs validados na borda.  
- **Acoplamento baixo**: dependências injetadas (construtor/fábrica). Sem singletons/globals.  
- **Coesão alta**: um adapter por tecnologia/responsabilidade; um use case por intenção.  
- **Observabilidade**: logs na borda/adapters; domínio permanece “mudo” (sem logging de framework).

## Arquitetura (camadas e componentes)
- **Domínio (puro)**: entidades, value objects, serviços de domínio, regras. **Sem imports de infra**.  
- **Aplicação (use cases)**: orquestra domínio; expõe **Inbound Ports**; depende de **Outbound Ports**.  
- **Ports**  
  - **Inbound Ports** (driven by outside): interfaces chamadas pela interface/transport (HTTP/CLI/Queue).  
  - **Outbound Ports** (driving outside): interfaces que a aplicação usa para falar com DB/HTTP/FS/etc.  
- **Adapters**  
  - **Primários** (para inbound): HTTP controllers/CLI/consumidores de fila → chamam **use cases**.  
  - **Secundários** (para outbound): implementam persistência, gateways externos, etc.  
- **Interface (borda)**: validação/serialização, mapeamento DTO↔domínio; sem regra de negócio.

> Manter **domínio isolado**. **Detalhes** (frameworks/DB/HTTP) ficam nas **bordas**.

## Convenções de Nomenclatura e Organização
- **Ports**: `XyzInPort` (entrada) e `XyzOutPort` (saída).  
- **Use cases**: `XyzUseCase` implementa `XyzInPort`.  
- **Adapters**:  
  - inbound: `HttpXyzController`, `CliXyzCommand`.  
  - outbound: `JpaXyzRepository`, `PrismaXyzRepo`, `S3FileStorage`.  
- **Pacotes/pastas sugeridos**  
  ```
  /domain
    /model        # entidades, VOs
    /service      # serviços de domínio
  /application
    /port/in      # inbound ports
    /port/out     # outbound ports
    /usecase
  /infrastructure
    /adapter/in   # controllers/handlers
    /adapter/out  # repos/gateways
    /config       # DI/wiring
  /interface      # DTOs/validators/mappers
  /tests
  ```
- **Mapeamento**: usar mappers explícitos `toDomain()/toDTO()` (sem anotações mágicas quando possível).

## Integração, Transações e Eventos
- **Transações**: iniciadas na borda do **use case**; adapters não iniciam transação por conta própria.  
- **Consistência**: preferir **transação por caso de uso**; para side-effects externos, usar **Outbox**/retentativas idempotentes.  
- **Eventos**: publicar **eventos de domínio** dentro do domínio (sincrono); propagar externos via adapter/event-bus.  
- **ACL (Anti-Corruption Layer)**: crie adapters tradutores quando integrar com sistemas legados/3rd.

## Instruções Específicas por Linguagem
- **Python**: tipos com `typing`; validação em DTOs (`pydantic` se já existe); `pytest`; `ruff` + `black`; DI leve por construtor/fábrica.  
- **TypeScript**: `strict` on; tipos primeiro; `vitest`/`jest`; `eslint` + `prettier`; inversão via construtor ou container simples (sem overengineering).  
- **Java**: JDK 17+; `record` para DTOs; JUnit 5; DI por construtor (Spring/Guice opcional); persistência via interfaces (ex.: `Repository`).

## Como quero que você responda (Copilot)
- **Sempre** começar com **plano numerado curto** (incremento mínimo).  
- Entregar **primeiro os testes** (domínio → use case com mocks → adapters integração).  
- **Comandos** para rodar testes/linters (ex.: `pytest -q`, `npm run test`, `mvn -q test`).  
- Encerrar com **checklist** (feito/pendências).  
- Se houver ambiguidade, **declare hipóteses** e siga com a solução mínima viável.

## Critérios de Teste por Camada
- **Domínio**: regras, invariantes, cenários de negócio (sem mocks).  
- **Aplicação**: fluxo do caso de uso, interação com **ports out** (mocks/fakes).  
- **Adapters**: integração real com tecnologia (DB, HTTP, FS) usando ambiente de teste/containers.  
- **Contratos**: testes de contrato entre **port out** e **adapter out** (ex.: pact/fixtures).

## Checklists para PR/Commit
- [ ] **Regra de Dependência** preservada (domínio não importa infra).  
- [ ] **Ports** pequenas e nomeadas pelo **propósito** (não pela tecnologia).  
- [ ] **Use cases** sem lógica de transporte/infra.  
- [ ] **Adapters** sem regra de negócio; mapeamentos claros DTO↔domínio.  
- [ ] Testes: domínio, aplicação (com mocks), adapters (integração) — todos verdes.  
- [ ] Funções/métodos ≤ ~30 linhas; nomes claros.  
- [ ] Dependências **injetadas**; zero singletons/globals.  
- [ ] Sem *dead code*; logs úteis nas bordas; erros significativos.  
- [ ] Documentados contratos de **ports** e decisões arquiteturais relevantes.

## Comandos (exemplos)
- **Python**: `ruff check . && black --check . && pytest -q`  
- **TypeScript**: `eslint . && prettier -c . && npm test`  
- **Java**: `mvn -q -DskipTests=false test`

---

### Dicas rápidas de aplicação
- Sempre comece pelo **use case + ports**; adie a escolha do adapter (DB/HTTP) até ter o contrato.  
- Se surgir uma nova tecnologia, **adicione um adapter**; **nunca** modifique o domínio para acomodá-la.  
- Quando uma regra “vaza” para o adapter, mova para serviço de domínio/use case.  
- Falhas externas → traduzir para **erros do domínio** nas **ports out** (ex.: `CustomerNotFound`).  
