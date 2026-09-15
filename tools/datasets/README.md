# Datasets multi-sujeito — imagetica motora membro superior

Todos os conversores padronizam para o **perfil OpenBCI 16ch @ 125 Hz**
(`Sample Index, EXG 0..15, ... , Annotations` com `T1`=esquerda, `T2`=direita,
`T0`=fim da tentativa). A Daisy 16ch real roda a 125 Hz (algumas GUIs
arredondam o rotulo para 128 Hz); o pipeline adaptativo
(`preprocess_window_adaptive`) aceita 125/128/250 Hz e qualquer n. de canais,
reamostrando/remapando para o canonico (250, 16).

## 1. PhysioNet EEGMMIDB (109 subj, 64ch @ 160 Hz)
So runs unilaterais punho esquerdo/direito (R03/R04/R07/R08/R11/R12).
```bash
.venv-dev/bin/python tools/datasets/physionet_eegmmidb.py --subjects 1-10 --runs imagined --out tools/datasets/data/physionet
```

## 2. BCI Competition IV 2a (9 subj, 22ch @ 250 Hz)
So mao esquerda/direita (769->T1, 770->T2; pe/lingua ignorados; T0 = cue+4s).
```bash
.venv-dev/bin/python tools/datasets/bnci_iv_2a_2b.py --dataset 2a --subjects 1-9 --out tools/datasets/data/bnci
```

## 3. BCI Competition IV 2b (9 subj, 3ch C3/Cz/C4 @ 250 Hz)
769->T1, 770->T2; demais canais zerados (documentado no `%Channel Mapping`).
```bash
.venv-dev/bin/python tools/datasets/bnci_iv_2a_2b.py --dataset 2b --subjects 1-9 --out tools/datasets/data/bnci
```

## 4. Treino generalizado (Group = dataset:sujeito)
```bash
.venv-dev/bin/python tools/datasets/train_generalized_all.py --data tools/datasets/data --epochs 30 --model-name generalized_mi_v1
```

## 5. Resultados reais (honestos, Leave-Subjects-Out)

Base: 72–200 arquivos, 15–33 grupos, 3 datasets. Validação sempre em
sujeitos NUNCA vistos no treino.

| modelo | val | obs |
|---|---|---|
| v1 CNN, 7 grupos | 0.53 | baseline |
| v2 CNN, 7 grupos | 0.633 | |
| v3 CNN + balance/augment | 0.616 | balance não ajudou o CNN |
| **v3 EEGNet (atual)** | **0.643** | melhor zero-shot; BNCI ~0.68, PhysioNet ~0.5 |
| v4 EEGNet dropout 0.4 + L2 1e-3 | 0.549 | regularização forte underfitou |
| v5 EEGNet 15 grupos + executados | 0.587 | |
| v6/v7 EEGNet, mapa corrigido | 0.50–0.54 | split com heldout 100% PhysioNet |
| v8 EEGNet + Euclidean Alignment | 0.542 | EA neutro aqui (z-score/janela já remove o shift que o EA alinha) |
| v9 ShallowConvNet (top DL no MOABB) | 0.545 | mesmo platô: teto é dos dados, não da arquitetura |

Lições:
- **Bug real encontrado e corrigido:** canais PhysioNet (`Fc5.`, `C3..`)
  caíam no fallback e embaralhavam a montagem (`common._norm_ch`).
- **Teto do zero-shot:** 0.55–0.65 conforme a loteria de sujeitos
  (S01: 0.82/fácil; S02: 0.32/anti-correlacionado). CNN, EEGNet,
  ShallowConvNet, com/sem Euclidean Alignment e com/sem balanceamento
  caem no mesmo platô — o limite está na transferência entre
  sujeitos/hardwares, não na arquitetura (consistente com MOABB 2024:
  Riemannian ≈ DL, e DL só brilha com 150+ trials/classe).
- **Calibração curta não ressuscita sujeito anti-correlacionado**
  (S02: 0.324 → 0.368 com 1 sessão). Sujeito fácil não precisa
  (S01: 0.824 → 0.838).
- **Infra de EA pronta:** `EASessionAligner` + flag no manifesto +
  calibração de 30 s na UI para futuros modelos com EA. ShallowConvNet
  adaptativa em `models.py` (`--arch shallow`).
- **Fluxo recomendado:** use o v3_eegnet como base no modo Livre e
  grave 2–3 sessões de Treino com a SUA touca/OpenBCI (mesmo hardware
  e montagem do uso real); o fine-tuning do app parte da base atual.
  Within-subject com mesmo hardware é o regime onde BCI-MI funciona
  (0.7–0.9 na literatura) — zero-shot cross-hardware é baseline.

Montagem: alvo `Fp1 Fp2 F7 F3 Fz F4 F8 T7 C3 Cz C4 T8 P7 P3 Pz P4`.
Quando o dataset nao tem o eletrodo, usa o motor mais proximo livre e
zera o restante (ver `%Channel Mapping` no CSV). Para experimentos clinicos,
audite a montagem real da touca antes de comparar sessoes.
