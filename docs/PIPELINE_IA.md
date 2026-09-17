# Pipeline EEG e IA

## Contrato atual

- Touca completa: 16 canais, 125 Hz (Daisy real; rotulos "128 Hz" de GUI sao
  aceitos e reamostrados), sem dividir canais por mao.
- Janela de inferencia: 250 amostras por 16 canais (2 s @ 125 Hz).
- Aquisicao e gravacao preservam EEG bruto; o filtro visual recebe uma copia.
- Treino e inferencia usam `infrastructure/ml/eeg_pipeline.py`: filtro Butterworth
  SOS de ordem 6, 8-30 Hz, `sosfiltfilt` sobre janela fechada e z-score por canal.
- O filtro usa a janela completa, nao e um filtro causal amostra a amostra.
- Caminho estrito (`preprocess_window`) rejeita shapes incorretos e valores nao
  finitos, sem padding de canais. Caminho adaptativo
  (`preprocess_window_adaptive`) aceita qualquer fs (125/128/250/...) e qualquer
  n. de canais, reamostrando/remapando para o canonico (250, 16) antes do filtro.
- Modelos `(None, 250, 16)` legados e `(None, None, 16)` adaptativos (GAP) sao
  aceitos; canais != 16 sao mapeados no preprocessing, nunca no peso.

## Modo Livre (sem VR obrigatorio)

- Nova tarefa "Livre": `FreeRunInferenceCoordinator` em loop continuo
  (janela 250, stride 125, ~1 predicao/s), sem `waiting_for_response` do VR.
- IA em tela sempre que EEG conectado; modelo/treino opcional (sem modelo, mostra
  so EEG ao vivo). Gravacao opcional com paciente.
- VR e ortese sao sinks opcionais com checkbox + cooldown 3 s + limiar 0.6:
  `send_udp_signal_free` / `send_esp32_signal_free`. Matriz valida: so VR,
  so ortese, ambos, nenhum.

## Calibracao obrigatoria + RL online

- Jogo e Livre exigem modelo proprio do paciente (pointer
  `patient_{id}.json`); sem ele, o app oferece o Treino de calibracao
  (minimo de trials T1/T2 configuravel, default 10, em Dev).
- Calibracao (padrao clinico: 10 trials T1/T2): fine-tune rede toda,
  LR 5e-5, 10 epocas, aumento leve no treino (2 replicas com ruido
  gaussiano pos-zscore + deslocamento temporal, sem channel-dropout)
  + peso de classe balanceado; validacao intacta. Ablacao offline
  (S012 mesmo paciente): 15ep/1e-4 overfita (val 0.46, TEST 0.53);
  10ep/5e-5 + aumento estabiliza (TEST 0.56 misto, 0.71 intramodal).
  Head-only nao move os pesos; valores em `runtime_config.get_runtime`.
  O dialogo mostra o delta base -> novo para mudanca visivel.
- RL: com `rl_enabled` (Dev, default off), botoes ✓/✗ rotulam a ultima
  predicao nos dois modos; no Jogo, CORRECT/WRONG do VR vira recompensa
  automatica (so com resposta esperada, anti-pacote-atrasado). A cada K
  feedbacks (default 5) um worker aplica clone-fit-swap (3 epocas, LR 5e-5,
  erro pesa 3x) + 1 replica com ruido leve por update (estabilidade no fim
  da sessao; S012: fim 0.68 -> 0.73, pico 0.79); limite de 20
  updates/sessao + "Restaurar base" (checkpoint
  pre-RL). Formalmente e aprendizado online supervisionado pelo feedback
  (em binario, o ✗ revela o rotulo) — mais estavel que policy-gradient no Qt.
  As janelas do feedback passam pelo mesmo pre-processamento do treino e
  da inferencia (Butterworth SOS ordem 6, 8-30 Hz + z-score por canal;
  bandpass -> EA -> z-score nos modelos com alinhamento).

## Ortesi (ESP32 CIMATEC)

- Protocolo 1-char @ 115200 baud (firmware `Ortese/ortese.cpp`):
  `l`=flexao IA, `e`=extensao IA, `o`=parar, `m`=modo IA, `z`=automatico,
  `p`=pausar, `r`=reset, `w/s/d/a`=calibracao. `\n` ignorado.
- Outputs de motor intactos (pinos 27/14/13/12, `TEMPO_MOVIMENTO 2s`,
  `TEMPO_PAUSA 1s`, `moverFlexao/Extensao/pararTudo`).
- BrainBridge nunca mais envia "LEFT"/"RIGHT"/"PING" (o `r` resetava e o `p`
  pausava a ortese).

## Dados

Novas gravacoes declaram o estagio RAW e timestamps de recepcao no logger.
Esses timestamps nao sao timestamps de aquisicao da placa. O emissor OpenBCI
precisa estar configurado a 125 Hz; o protocolo atual nao comprova a taxa real.

O conversor EDF reamostra explicitamente para 125 Hz antes de extrair dados e
calcular indices de anotacoes. CSVs legados sem metadados emitem avisos e ficam
com proveniencia desconhecida: sua aceitacao nao comprova que sejam brutos ou
que tenham a montagem correta. Audite esses arquivos antes de experimentos.

## Modelos e compatibilidade

Novos checkpoints possuem um sidecar `.pipeline.json` com o contrato de
processamento. Checkpoints antigos sem esse contrato nao podem ser usados na
inferencia RAW nem reinterpretados automaticamente para fine-tuning.

A selecao automatica ignora bases legadas/incompativeis com mensagem explicita;
sem base compativel, treina do zero. Uma referencia publicada corrompida gera
erro em vez de trocar silenciosamente de modelo. Checkpoints anteriores sao
preservados. O candidato ainda e carregado automaticamente apos sucesso:
aprovacao operacional, rollback e calibracao de confianca permanecem pendentes.

## Avaliacao

O treinamento especifico separa trials entre treino e validacao. Janelas
sobrepostas do mesmo trial nao cruzam particoes. Sao necessarios ao menos dois
trials distintos de cada classe. Copias identicas de um arquivo conservam a
identidade usada no agrupamento.

O treinamento generalizado separa participantes e exige ambas as classes em
cada particao. A validacao usada para early stopping nao e um teste independente.
Ainda e necessario um teste posterior por sessao/paciente, sem exposicao previa
do checkpoint, para medir generalizacao e beneficio da personalizacao.

### Estado medido (multi-dataset, Leave-Subjects-Out)

Modelo atual: `generalized_left_right_eegmmidb_20260518_231607` (CNN 1D,
16ch @ 125 Hz, unico .keras em `brainbridge_v2/infrastructure/data/models`).
Zero-shot medido em 17/09 (S011/S012/S016, 728 janelas, `metrics_zeroshot.json`):
**media 0.64** (S011 0.62, S012 0.61, S016 0.68). Candidatos removidos na
mesma medicao: `metrics_generalized_cnn` 0.54, `modelo_full` 0.49 (chance).
Detalhes e tentativas (v1–v7) em `tools/datasets/README.md`.
Variância entre sujeitos domina (0.32–0.82 por sujeito/arquivo); por isso
o fluxo recomendado é base generalizada + 2–3 sessões de Treino no
hardware/montagem real (fine-tuning parte do checkpoint mais recente).

## Limites

- Posicoes e ordem fisica dos eletrodos ainda precisam ser confirmadas.
- Movimento real do lado nao afetado e imaginado do lado afetado devem ser
  analisados separadamente na avaliacao do protocolo.
- A inferencia roda em worker Qt, com um unico job pendente e descarte logico
  de resultados antigos. Carregamento/warmup ainda podem bloquear a interface;
  uma chamada TensorFlow em execucao nao e interrompida pelo cancelamento.
- Testes sinteticos/fakes nao demonstram melhoria de acuracia nem eficacia clinica.
- Protocolos Unity e serial permanecem inalterados.

## Testes seguros

Execute a suite `tests/unit/layers`, incluindo testes numericos do pipeline,
segmentacao/splits, publicacao dos manifestos, aquisicao RAW e conversao EDF.
Em ambiente com as dependencias de teste instaladas:

```bash
python -m pytest -o addopts='' tests/unit/layers -q
```

Nao execute indiscriminadamente a suite raiz: ela tambem contem diagnosticos
legados que podem acessar dispositivos fisicos.
