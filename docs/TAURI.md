# BrainBridge Web + Tauri (Fases 1–3)

## Arquitetura

```
webui/                  React + Vite (Fase 2) — dist servido pelo Tauri
webui/src-tauri/        App Tauri: janela nativa + sidecar Python (Fase 3)
brainbridge_v2/server/  Backend FastAPI + WebSocket (Fase 1), sem Qt
```

O Python continua com 100% da regra de negócio (mesmos controllers da UI
PyQt). O frontend fala REST (`/api/*`, docs em `/docs`) + WebSocket
(`/ws/eeg`). Sem internet: tudo em `localhost`.

## Rodar em dev (2 terminais)

```bash
# 1. backend (venv do repo)
.venv-dev/bin/python -m brainbridge_v2.server --port 8000

# 2. frontend
cd webui && npm install && npm run dev   # http://localhost:5173
```

Ou tudo junto como app nativo:

```bash
cd webui && npx tauri dev
```

O `tauri dev` abre a janela do app, sobe o Vite e lança o backend como
sidecar (`src-tauri/binaries/brainbridge-server-*`, script dev que usa o
venv do repo).

## Estrutura da UI web

- `src/api.js` — contrato REST/WS. Dentro do Tauri (`window.__TAURI_INTERNALS__`),
  a base vira `http://127.0.0.1:8000` automaticamente; fora, mesma origem (proxy Vite).
- `src/useEeg.js` — hook do stream `/ws/eeg` com reconexão por `enabled`.
- `src/components/Streaming.jsx` — cartões + plot canvas 16ch + sidebar IA/RL/marcadores;
  loop online (janela 250/stride 125), espelho VR/órtese, trava de calibração, fluxo de treino.
- `Patients.jsx`, `Training.jsx` (treino + SSE + modelos), `Settings.jsx` (runtime_config + RL).

## Build de produção

```bash
cd webui && npm run build          # gera dist/
npx tauri build                    # .deb/.AppImage em src-tauri/target/release/bundle/
```

O bundle final precisa do **binário PyInstaller** como sidecar
(`src-tauri/binaries/brainbridge-server-<target>`), substituindo o script
dev. Gerar com `brainbridge_server.spec` (a criar a partir de
`brainbridge.spec`) e renomear com o sufixo do target. Sem o binário, o
`tauri build` empacota só o frontend (o app abre, mas sem backend).

## Notas

- `tauri.conf.json`: `frontendDist ../dist`, `devUrl` Vite, CSP nulo
  (localhost), `externalBin: ["binaries/brainbridge-server"]`, categoria Medical.
- Permissão mínima em `capabilities/default.json`: só `shell:allow-spawn`
  do sidecar. Nada de FS/dialog/notification (não usados).
- Ícones gerados via `npx tauri icon app-icon.png --output src-tauri/icons`.
