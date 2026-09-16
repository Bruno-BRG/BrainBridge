# BrainBridge interface — sistema visual

Estação atlas de neurofisiologia: casca clara em papel mineral quente, campo de sinal escuro para o EEG ao vivo, regras hairline, cantos de 8px (6px em controles), cores chapadas sem degradê, tipografia Roboto com numerais tabulares. Ícones Lucide em traço único; nada de emoji na interface.

## Tokens

- Papel `--paper #f4f1ea`, superfície `--surface #fffdf8`, tinta `--ink #1c2530`.
- Acentos com função fixa: cobalto `#2547c9` (ação/seleção), jade `#1d8f60` (pronto/ativo), vermelhão `#c93a3a` (parar/erro), âmbar (avisos).
- Poço de sinal `--signal #091827`; o canvas usa o mesmo fundo para fusão total.
- Fonte: pilha system-ui compacta; mono só para caminhos, índices e logs.

## Composição

- Trilho esquerdo escuro com identidade, 4 rotas, status do backend e versão.
- Sessão: faixa de prontidão em 3 passos, banda de preparação (paciente, conexões, gravação), painel de sinal 2fr + trilho de interpretação 348px.
- Pacientes: diretório com busca, detalhe clínico e formulário de cadastro.
- Treino: fluxo em 3 passos com validação de trials e biblioteca de modelos.
- Ajustes: parâmetros agrupados por calibração e RL, resumo de pendências e painel de checkpoint.

## Estados e movimento

- Prontidão avança em chips tracejados que acendem em jade; gravação pulsa em vermelhão.
- Botão de gravação isolado: cobalto para iniciar, vermelhão para encerrar, sem mudar de lugar.
- Transições de 150–300ms só para feedback de estado; respeita `prefers-reduced-motion`.
- Vazios ensinam o próximo passo; avisos nomeiam problema e recuperação.

## Regras duras

- Nenhuma funcionalidade ou chamada de API some no redesign; texto em pt-BR.
- Estados nunca dependem só de cor: sempre há ícone ou rótulo junto.
- Contraste AA no texto; foco sempre visível.
- Mobile: trilho vira linha rolável, painéis empilham, segmentados quebram em 2 colunas.
- Sessão ao vivo cabe inteira na viewport sem scroll: faixa de preparação compacta, canvas flexível, trilho com rolagem interna só como válvula de escape.
- Sessão sem scroll por construção: cadeia flex com shrink liberado, canvas com piso de 120px, trilho com rolagem interna.
- EEG com faixas alternadas, divisórias entre canais, rótulos C01–C16 nas duas bordas e escala vertical selecionável (±50/±100/±250 μV).
- Painel de feedback sempre fixo e visível; botões desabilitados sem leitura; sem pop-ups.
- Linguagem direta nas pistas da tarefa (Esquerda/Direita, sem T1/T2 na face dos botões).
- Painéis sensíveis ao contexto: feedback de acerto só aparece após uma predição; placar e acurácia VR só no modo Jogo.
- Interpretação sem microcopy: só estado, probabilidades, modelo, espelhamento e placar quando aplicável.
- Biblioteca mostra só o nome do modelo; caminho completo fica no tooltip.
- Ajustes misturam config do backend com preferências locais (dispositivos e exibição, guardadas no navegador).
- Conexões falam a verdade: EEG mostra modo real (udp/synth) e falha alto sem fallback; órtese mostra porta e motivo da falha; baseline mostra regressiva; jogo publica sessão no VR.
