import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Activity, ArrowDownLeft, ArrowRight, ArrowUpRight, BrainCircuit, Check,
  Gamepad2, HeartPulse, Link2, Play, Radio, RefreshCw, RotateCcw,
  Square, Usb, Waves, X,
} from 'lucide-react';
import { api } from '../api.js';
import { useEegStream } from '../useEeg.js';
import { Notice, PanelHeading, StatusChip, loadLocalSettings, saveLocalSettings } from './ui.jsx';

const CHANNEL_COLORS = [
  '#63b3ed', '#48bb78', '#f6ad55', '#fc8181', '#b794f4', '#4fd1c5',
  '#f687b3', '#90cdf4', '#68d391', '#fbd38d', '#9ae6b4', '#feb2b2',
  '#d6bcfa', '#81e6d9', '#fbb6ce', '#bee3f8',
];
const WINDOW = 250; // 2 s @125Hz
const STRIDE = 125;
const PLOT_BUFFER_SECONDS = 30;

function EegCanvas({ bufferRef, seconds, scale }) {
  const viewSeconds = Math.min(Math.max(Number(seconds) || 8, 2), PLOT_BUFFER_SECONDS);
  const fullScale = Math.max(Number(scale) || 100, 10);
  const canvasRef = useRef(null);

  useEffect(() => {
    let frame = 0;
    const draw = () => {
      frame = requestAnimationFrame(draw);
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      const width = Math.max(50, Math.floor(rect.width * ratio));
      const height = Math.max(50, Math.floor(rect.height * ratio));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#091827';
      ctx.fillRect(0, 0, width, height);
      const buf = bufferRef.current;
      const n = buf.length;
      if (n < 2) return;
      const view = Math.min(n, viewSeconds * 125);
      const start = n - view;
      const laneH = height / 16;
      const yOf = (ch, v) => (ch + 0.5) * laneH - (v / fullScale) * (laneH * 0.45);
      ctx.lineWidth = Math.max(1, ratio * 0.8);
      // faixas alternadas por canal
      ctx.fillStyle = 'rgba(160, 200, 220, 0.035)';
      for (let ch = 0; ch < 16; ch += 2) {
        ctx.fillRect(0, ch * laneH, width, laneH);
      }
      // divisórias entre canais
      ctx.strokeStyle = 'rgba(150, 190, 205, 0.32)';
      ctx.beginPath();
      for (let i = 0; i <= 16; i += 1) {
        const y = Math.round(i * laneH) + 0.5;
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
      }
      ctx.stroke();
      // linha central de cada canal
      ctx.strokeStyle = 'rgba(150, 190, 205, 0.12)';
      ctx.beginPath();
      for (let ch = 0; ch < 16; ch += 1) {
        const y = Math.round((ch + 0.5) * laneH) + 0.5;
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
      }
      ctx.stroke();
      // rótulos nas duas bordas
      ctx.fillStyle = 'rgba(170, 205, 220, 0.85)';
      ctx.font = `${10 * ratio}px ui-monospace, monospace`;
      for (let ch = 0; ch < 16; ch += 1) {
        const y = (ch + 0.5) * laneH + (4 * ratio);
        const left = `C${String(ch + 1).padStart(2, '0')}`;
        ctx.fillText(left, 5 * ratio, y);
        const right = String(ch + 1);
        ctx.fillText(right, width - (14 * ratio) - ctx.measureText(right).width, y);
      }
      for (let ch = 0; ch < 16; ch += 1) {
        ctx.strokeStyle = CHANNEL_COLORS[ch];
        ctx.beginPath();
        // Decimação por coluna de pixel (min/max): tira o serrilhado e
        // reduz o trabalho por quadro sem mudar os dados.
        const cols = Math.max(1, Math.floor(width));
        let started = false;
        for (let c = 0; c < cols; c += 1) {
          const i0 = start + Math.floor((c / cols) * view);
          const i1 = Math.min(n, start + Math.floor(((c + 1) / cols) * view));
          let mn = Infinity;
          let mx = -Infinity;
          for (let i = i0; i < i1; i += 1) {
            const sample = buf[i];
            if (!sample) continue;
            const v = sample[ch] || 0;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
          }
          if (mn === Infinity) continue;
          const x = ((c + 0.5) / cols) * width;
          if (!started) {
            ctx.moveTo(x, yOf(ch, (mn + mx) / 2));
            started = true;
          } else {
            ctx.lineTo(x, yOf(ch, mx));
            ctx.lineTo(x, yOf(ch, mn));
          }
        }
        ctx.stroke();
      }
    };
    frame = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(frame);
  }, [bufferRef, viewSeconds, fullScale]);

  return <canvas ref={canvasRef} style={{ flex: 1, width: '100%', minHeight: 0 }} />;
}

export default function Streaming({ active }) {
  const [local, setLocal] = useState(loadLocalSettings);
  const [patients, setPatients] = useState([]);
  const [patientId, setPatientId] = useState('');
  const [task, setTask] = useState('livre');
  const [eegOn, setEegOn] = useState(false);
  const [devices, setDevices] = useState(null);
  const [model, setModel] = useState(null);
  const [markers, setMarkers] = useState({ t1_count: 0, t2_count: 0 });
  const [recording, setRecording] = useState(null); // {id, sessionId, csvPath, startedAt}
  const [elapsed, setElapsed] = useState(0);
  const [baselineLeft, setBaselineLeft] = useState(null); // segundos restantes ou null
  const [result, setResult] = useState(null); // {side, conf, left, right}
  const [accuracy, setAccuracy] = useState({ correct: 0, total: 0 });
  const [placar, setPlacar] = useState({ left: 0, right: 0 });
  const [rl, setRl] = useState({ enabled: false, updates_applied: 0 });
  const [rlLabeled, setRlLabeled] = useState(0);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const bufferRef = useRef([]); // amostras 16ch (plot)
  const windowRef = useRef([]); // janela p/ inferencia
  const inferBusy = useRef(false);
  const feedbackBuf = useRef([]); // {window, pred, label?}
  const appliedRef = useRef(0);
  const lastPred = useRef(null);
  const [mirrorVr, setMirrorVr] = useState(true);
  const [mirrorOrt, setMirrorOrt] = useState(true);

  const rlRef = useRef(rl);
  rlRef.current = rl;
  const vrSeqRef = useRef(0);

  useEffect(() => {
    const reload = () => setLocal(loadLocalSettings());
    window.addEventListener('bb:local-settings', reload);
    return () => window.removeEventListener('bb:local-settings', reload);
  }, []);

  useEffect(() => {
    if (active) setLocal(loadLocalSettings());
  }, [active]);

  const say = (text) => {
    setMessage(text);
    setError('');
  };
  const fail = (err) => {
    setError(err && err.message ? err.message : String(err));
  };

  const refreshAll = useCallback(async () => {
    try {
      const [list, devs, loaded, rlStatus, markerState] = await Promise.all([
        api.patients(), api.devices(), api.loadedModel(), api.rlStatus(), api.markerState(),
      ]);
      setPatients(list);
      setDevices(devs);
      setModel(loaded);
      setRl({ enabled: !!rlStatus.enabled, updates_applied: rlStatus.updates_applied || 0 });
      setMarkers({ t1_count: markerState.t1_count || 0, t2_count: markerState.t2_count || 0 });
    } catch (err) {
      fail(err);
    }
  }, []);

  useEffect(() => {
    refreshAll();
    const timer = setInterval(() => {
      api.devices().then(setDevices).catch(() => {});
      api.rlStatus().then((s) => setRl({ enabled: !!s.enabled, updates_applied: s.updates_applied || 0 })).catch(() => {});
    }, 3000);
    return () => clearInterval(timer);
  }, [refreshAll]);

  // inferencia continua (modo online): janela 250, stride 125
  const runInference = useCallback(async (window) => {
    if (inferBusy.current || !model) return;
    inferBusy.current = true;
    try {
      const pred = await api.predict(window.map((row) => Array.from(row)));
      const side = pred.predicted_index === 0 ? 'ESQUERDA' : 'DIREITA';
      setResult({ side, conf: pred.confidence, left: pred.left_probability, right: pred.right_probability });
      setPlacar((p) => pred.predicted_index === 0
        ? { left: p.left + 1, right: p.right }
        : { left: p.left, right: p.right + 1 });
      lastPred.current = { window: window.map((row) => Array.from(row)), pred: pred.predicted_index };
      feedbackBuf.current.push({ window: window.map((row) => Array.from(row)), pred: pred.predicted_index, label: null });
      if (feedbackBuf.current.length > 200) feedbackBuf.current.shift();
      // espelhamento opcional VR/ortese
      const direction = pred.predicted_index === 0 ? 'esquerda' : 'direita';
      if (mirrorVr && devices && devices.unity && devices.unity.server_active) {
        api.unityAction(direction).catch(() => {});
      }
      if (mirrorOrt && devices && devices.esp32 && devices.esp32.connected) {
        api.esp32Action(direction).catch(() => {});
      }
    } catch {
      /* sem modelo ou erro transitório: só EEG ao vivo */
    } finally {
      inferBusy.current = false;
    }
  }, [model, devices, mirrorVr, mirrorOrt]);

  const onBatch = useCallback((batch) => {
    const buf = bufferRef.current;
    for (const sample of batch) {
      buf.push(sample);
      windowRef.current.push(sample);
    }
    const cap = PLOT_BUFFER_SECONDS * 125 + 250;
    if (buf.length > cap) buf.splice(0, buf.length - cap);
    if (windowRef.current.length >= WINDOW) {
      const window = windowRef.current.slice(-WINDOW);
      windowRef.current = windowRef.current.slice(-WINDOW + STRIDE);
      runInference(window);
    }
  }, [runInference]);

  const connectOpts = {
    host: local.eegHost || 'localhost',
    port: Number(local.eegPort) || 12345,
    simulate: local.eegSimulate !== false,
  };
  const connectOptsWithStream = {
    ...connectOpts,
    stream: local.eegStream === 'filtered' ? 'filtered' : 'raw',
  };
  const eegFail = useCallback((err) => {
    setEegOn(false);
    api.eegDisconnect().catch(() => {});
    fail(new Error(`EEG desconectado: ${err && err.message ? err.message : err}`));
  }, []);
  const { status: eegStatus } = useEegStream(connectOptsWithStream, { enabled: eegOn, onBatch, onError: eegFail });
  const eegConnectAt = useRef(0);

  // Watchdog: conectado ao OpenBCI mas sem pacotes = erro crítico, sem fallback.

  useEffect(() => {
    if (!recording) return undefined;
    const timer = setInterval(() => setElapsed(Math.floor((Date.now() - recording.startedAt) / 1000)), 1000);
    return () => clearInterval(timer);
  }, [recording]);

  useEffect(() => {
    if (!recording || task !== 'baseline') return undefined;
    const timer = setInterval(async () => {
      try {
        const st = await api.baselineTick();
        const s = (st && st.state) || st || {};
        if (s.baseline_remaining_seconds != null) {
          setBaselineLeft(Number(s.baseline_remaining_seconds));
        }
        if (st && st.finished) {
          setBaselineLeft(0);
          say('Baseline concluído.');
        }
      } catch {
        /* mantém a última contagem visível */
      }
    }, 1000);
    return () => clearInterval(timer);
  }, [recording, task]);

  const toggleEeg = async () => {
    if (eegOn) {
      setEegOn(false);
      eegConnectAt.current = 0;
      try {
        await api.eegDisconnect();
      } catch (err) {
        fail(err);
      }
    } else {
      setError('');
      setMessage('');
      eegConnectAt.current = Date.now() / 1000;
      setEegOn(true);
    }
  };

  const toggleEsp32 = async () => {
    setError('');
    setMessage('');
    try {
      if (esp32.connected) {
        await api.esp32Disconnect();
      } else {
        const res = await api.esp32Connect(local.esp32Port || undefined);
        if (res && res.connected) {
          say(`Órtese conectada em ${res.port || 'porta detectada'}.`);
        } else {
          fail(new Error(res && res.reason ? `Órtese: ${res.reason}` : 'Órtese não conectou.'));
        }
      }
    } catch (err) {
      fail(err);
    } finally {
      refreshAll();
    }
  };

  const toggleRecording = async () => {
    setError('');
    setMessage('');
    if (!recording) {
      if (!patientId) {
        fail(new Error('Selecione um paciente.'));
        return;
      }
      // trava de calibração p/ jogo/livre
      if (task === 'jogo' || task === 'livre') {
        try {
          const cal = await api.calibration(Number(patientId));
          if (!cal.calibrated) {
            if (!window.confirm(
              `Paciente sem calibração (mínimo ${cal.trials_required} trials). Ir para o Treino agora?`)) return;
            setTask('treino');
            return;
          }
        } catch (err) {
          fail(err);
          return;
        }
      }
      try {
        const rec = await api.recordingStart({
          patient_id: Number(patientId), filename: `web_${task}.csv`, task_type: task,
        });
        let sessionId = null;
        try {
          const sess = await api.sessionStart({
            patient_id: Number(patientId), task_type: task, recording_id: rec.id,
          });
          sessionId = sess.recording_id ? sess.id : sess.id;
        } catch {
          sessionId = null;
        }
        setRecording({ id: rec.id, sessionId, csvPath: rec.csv_path, startedAt: Date.now() });
        setElapsed(0);
        setBaselineLeft(null);
        if (task === 'baseline') {
          try {
            const st = await api.baselineStart(300);
            const s = (st && st.state) || st || {};
            if (s.baseline_remaining_seconds != null) {
              setBaselineLeft(Number(s.baseline_remaining_seconds));
            } else {
              setBaselineLeft(300);
            }
          } catch (err) {
            fail(new Error(`Baseline não iniciou: ${err.message}`));
          }
        }
        if (devices && devices.unity
            && devices.unity.server_active && devices.unity.client_connected) {
          try {
            await api.unityPublishSession(Number(patientId), task);
            await api.unityTrigger();
          } catch (err) {
            say(`Gravando, mas o VR não respondeu: ${err.message}`);
          }
        }
        say(task === 'treino' ? 'Gravando treino…' : 'Gravando…');
      } catch (err) {
        fail(err);
      }
    } else {
      try {
        const stopped = await fetch(`/api/recordings/${recording.id}/stop`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ duration_seconds: elapsed }),
        }).then((r) => r.json()).then((b) => b.data);
        try {
          await api.sessionEnd();
        } catch {
          /* sem sessão ativa */
        }
        const wasTask = task;
        const csvPath = stopped && stopped.csv_path;
        setRecording(null);
        setBaselineLeft(null);
        try {
          await api.unityEndTask();
          if (wasTask === 'jogo') {
            await api.unityEndSession('Parabéns! Sessão finalizada com sucesso!');
          }
        } catch {
          /* VR pode já estar fora do ar; a gravação foi salva mesmo assim */
        }
        if (wasTask === 'treino' && csvPath) {
          await startTrainingFlow(csvPath);
        } else {
          say('Gravação finalizada.');
        }
      } catch (err) {
        fail(err);
      }
    }
  };

  const startTrainingFlow = async (csvPath) => {
    try {
      say('Treino gravado. Verificando trials…');
      const check = await api.trainingCheck(csvPath);
      if (!check.ok) {
        const go = window.confirm(
          `Sessão com ${check.trials} trials (mínimo ${check.required}). Treinar assim mesmo?`);
        if (!go) {
          say('Treino descartado para calibração.');
          return;
        }
      }
      say('Treinando modelo…');
      const job = await api.trainingStart(csvPath, Number(patientId), true);
      for (;;) {
        await new Promise((r) => setTimeout(r, 2000));
        const st = await api.trainingStatus(job.job_id);
        if (st.status === 'done') {
          const lines = (st.messages || []).slice(-4).join('\n');
          say(`Treino concluído.\n${lines}`);
          const loaded = await api.loadedModel();
          setModel(loaded);
          break;
        }
        if (st.status === 'error') {
          fail(new Error(st.error || 'Falha no treino.'));
          break;
        }
        const last = (st.messages || []).slice(-1)[0];
        if (last) say(`Treinando… ${last}`);
      }
    } catch (err) {
      fail(err);
    }
    refreshAll();
  };

  const sendMarker = async (marker) => {
    try {
      const reg = await api.marker(marker, task);
      if (reg.accepted) {
        const st = await api.markerState();
        setMarkers({ t1_count: st.t1_count || 0, t2_count: st.t2_count || 0 });
      }
    } catch (err) {
      fail(err);
    }
  };

  const sendFeedback = async (correct, via = 'manual') => {
    const pending = [...feedbackBuf.current].reverse().find((f) => f.label === null);
    if (!pending) {
      if (via === 'manual') fail(new Error('Sem predição recente para rotular.'));
      return false;
    }
    pending.label = correct ? pending.pred : 1 - pending.pred;
    const labeled = feedbackBuf.current.filter((f) => f.label !== null).length;
    setRlLabeled(labeled);
    if (via === 'manual') say(correct ? 'Feedback: acertou ✓' : 'Feedback: errou ✗');
    // aplica sozinho a cada K feedbacks
    try {
      const cfg = await api.config();
      const batchK = Number(cfg.rl_batch_k || 5);
      const fresh = feedbackBuf.current.filter((f) => f.label !== null).slice(appliedRef.current);
      if (rlRef.current.enabled && fresh.length >= batchK) {
        appliedRef.current += fresh.length;
        const weights = fresh.map((f) => (f.label === f.pred ? 1.0 : Number(cfg.rl_mistake_weight || 3.0)));
        say(`RL: aplicando update com ${fresh.length} feedbacks…`);
        const out = await api.rlUpdate({
          windows: fresh.map((f) => f.window),
          labels: fresh.map((f) => f.label),
          weights,
          epochs: Number(cfg.rl_epochs || 3),
          lr: Number(cfg.rl_lr || 0.00005),
        });
        const st = await api.rlStatus();
        setRl({ enabled: !!st.enabled, updates_applied: st.updates_applied || 0 });
        say(`RL aplicado (n=${out.n}).`);
      }
    } catch (err) {
      fail(err);
    }
    return true;
  };

  // No modo jogo, os veredictos CORRECT/WRONG chegam do VR sozinhos:
  // alimentam a acurácia e rotulam a predição pendente sem clique.
  const pollVrVerdicts = useCallback(async () => {
    try {
      const data = await api.unityVerdicts(vrSeqRef.current);
      vrSeqRef.current = data.last_seq || vrSeqRef.current;
      for (const v of data.verdicts || []) {
        const correct = v.verdict === 'correct';
        setAccuracy((a) => ({ correct: a.correct + (correct ? 1 : 0), total: a.total + 1 }));
        const labeled = await sendFeedback(correct, 'vr');
        say(labeled
          ? (correct ? 'VR: acertou ✓' : 'VR: errou ✗')
          : (correct ? 'VR: acertou ✓ (sem predição pendente)' : 'VR: errou ✗ (sem predição pendente)'));
      }
    } catch {
      /* backend ou VR indisponível: tenta de novo no próximo ciclo */
    }
  }, []);

  useEffect(() => {
    if (task !== 'jogo') return undefined;
    const timer = setInterval(pollVrVerdicts, 1500);
    return () => clearInterval(timer);
  }, [task, pollVrVerdicts]);

  const restoreBase = async () => {
    try {
      await api.rlRestore();
      feedbackBuf.current = [];
      appliedRef.current = 0;
      setRlLabeled(0);
      say('Modelo restaurado para o checkpoint pré-RL.');
    } catch (err) {
      fail(err);
    }
  };

  const eeg = devices && devices.eeg ? devices.eeg : {};
  const unity = devices && devices.unity ? devices.unity : {};
  const esp32 = devices && devices.esp32 ? devices.esp32 : {};
  const TASK_LABELS = { baseline: 'Baseline', jogo: 'Jogo', livre: 'Livre', treino: 'Treino' };
  const patientName = (patients.find((p) => String(p.id) === String(patientId)) || {}).name || '—';
  const fmtElapsed = `${String(Math.floor(elapsed / 3600)).padStart(2, '0')}:${String(Math.floor((elapsed % 3600) / 60)).padStart(2, '0')}:${String(elapsed % 60).padStart(2, '0')}`;
  const accRate = accuracy.total > 0 ? (100 * accuracy.correct) / accuracy.total : 0;
  // Watchdog: UDP ligado mas sem pacotes do OpenBCI = erro crítico, sem fallback.
  const nowEpoch = Date.now() / 1000;
  const packetAge = eeg.last_packet_at ? nowEpoch - Number(eeg.last_packet_at) : null;
  const eegStarved = Boolean(
    eegOn && eeg.mode === 'udp'
    && (packetAge == null
      ? (eegConnectAt.current > 0 && nowEpoch - eegConnectAt.current > 3)
      : packetAge > 3),
  );
  const patientReady = Boolean(patientId);
  const signalReady = Boolean(eegOn && eegStatus.running);
  const sessionReady = patientReady && eegOn;
  const eegScale = Number(local.eegScale) || 100;
  const setEegScale = (value) => setLocal(saveLocalSettings({ eegScale: value }));

  return (
    <div className="stream-layout">
      <section className="session-prep surface">
        <div className="readiness-rail" aria-label="Prontidão da sessão">
          <span className={patientReady ? 'ready' : ''}><i>{patientReady ? <Check size={12} /> : '1'}</i>Paciente</span>
          <span className={signalReady ? 'ready' : ''}><i>{signalReady ? <Check size={12} /> : '2'}</i>Sinal EEG</span>
          <span className={recording ? 'ready recording' : ''}><i>{recording ? <Radio size={12} /> : '3'}</i>Gravação</span>
        </div>

        <div className="prep-grid">
          <div className="prep-block patient-block">
            <div className="micro-label"><HeartPulse size={14} /> Paciente e protocolo</div>
            <div className="field-with-action">
              <label className="sr-only" htmlFor="session-patient">Paciente</label>
              <select id="session-patient" value={patientId} onChange={(e) => setPatientId(e.target.value)}>
                <option value="">Selecione o paciente</option>
                {patients.map((p) => <option key={p.id} value={p.id}>{p.name} · ID {p.id}</option>)}
              </select>
              <button className="icon-button" onClick={refreshAll} title="Atualizar dados" aria-label="Atualizar dados"><RefreshCw size={17} /></button>
            </div>
            <div className="segmented-control" aria-label="Protocolo da sessão">
              {['baseline', 'jogo', 'livre', 'treino'].map((t) => <button key={t} className={task === t ? 'active' : ''} onClick={() => setTask(t)}>{t === 'baseline' ? 'Baseline' : t === 'jogo' ? 'Jogo' : t === 'livre' ? 'Livre' : 'Treino'}</button>)}
            </div>
          </div>

          <div className="prep-block connections-block">
            <div className="micro-label"><Link2 size={14} /> Conexões</div>
            <div className="connection-list">
              <div className="connection-row"><span><i className={eeg.running ? 'device-dot on' : 'device-dot'} /><Waves size={17} /><b>EEG</b><small>{eeg.running ? `${eeg.mode || 'ativo'} · ${eeg.samples || 0}` : 'standby'}</small></span><button className="compact-button" onClick={toggleEeg}>{eegOn ? 'Desconectar' : 'Conectar'}</button></div>
              <div className="connection-row"><span><i className={unity.server_active ? 'device-dot on' : 'device-dot'} /><Gamepad2 size={17} /><b>VR</b><small>{unity.client_connected ? 'cliente conectado' : 'sem cliente'}</small></span><label className="mirror-mini" title="Enviar cada leitura ao VR"><input type="checkbox" checked={mirrorVr} onChange={(e) => setMirrorVr(e.target.checked)} /><span className="switch tiny" />Espelhar</label><button className="compact-button" onClick={() => (unity.server_active ? api.unityStop() : api.unityStart()).then(refreshAll).catch(fail)}>{unity.server_active ? 'Parar' : 'Iniciar'}</button></div>
              <div className="connection-row"><span><i className={esp32.connected ? 'device-dot on' : 'device-dot'} /><Usb size={17} /><b>Órtese</b><small>{esp32.connected ? `conectada · ${esp32.port || ''}` : 'desconectada'}</small></span><label className="mirror-mini" title="Enviar cada leitura à órtese"><input type="checkbox" checked={mirrorOrt} onChange={(e) => setMirrorOrt(e.target.checked)} /><span className="switch tiny" />Espelhar</label><button className="compact-button" onClick={toggleEsp32}>{esp32.connected ? 'Soltar' : 'Conectar'}</button></div>
            </div>
          </div>

          <div className="prep-block record-block">
            <div className="micro-label"><Radio size={14} /> Gravação</div>
            <button className={`record-button ${recording ? 'is-recording' : ''}`} onClick={toggleRecording} disabled={!eegOn && !recording}>
              <span className="record-icon">{recording ? <Square size={17} fill="currentColor" /> : <Play size={18} fill="currentColor" />}</span>
              <span><strong>{recording ? `Encerrar ${TASK_LABELS[task] || task}` : `Iniciar ${TASK_LABELS[task] || task}`}</strong><small>{recording ? 'Sessão em andamento' : sessionReady ? 'Tudo pronto para começar' : 'Conecte o EEG e selecione o paciente'}</small></span>
            </button>
            <div className="record-facts"><span><small>Estado</small><strong>{recording ? 'Gravando' : 'Parado'}</strong></span><span><small>Paciente</small><strong>{patientName}</strong></span>{task === 'baseline' && recording && baselineLeft != null && <span><small>Baseline restante</small><strong className="tabular">{`${String(Math.floor(baselineLeft / 60)).padStart(2, '0')}:${String(baselineLeft % 60).padStart(2, '0')}`}</strong></span>}<span><small>Duração</small><strong className="tabular">{fmtElapsed}</strong></span></div>
          </div>
        </div>
      </section>

      <Notice type="error">{error}</Notice>
      <Notice type="success">{message}</Notice>
      {eegStarved && <Notice type="error">{`Sem dados do OpenBCI há mais de 3s (${eeg.host || local.eegHost || 'localhost'}:${eeg.port || local.eegPort || 12345}, stream ${eeg.stream || local.eegStream || 'raw'}). Verifique o OpenBCI GUI. Nada foi gravado em modo sintético.`}</Notice>}

      <div className="live-workspace">
        <section className="signal-panel">
          <PanelHeading icon={Activity} title="Sinal EEG ao vivo" action={<div className="panel-badges"><StatusChip>16 canais</StatusChip><StatusChip>125 Hz</StatusChip><StatusChip tone={eegStatus.running ? 'positive' : ''} dot>{eegStatus.running ? eegStatus.mode : 'Aguardando'}</StatusChip></div>} />
          <div className="signal-canvas"><EegCanvas bufferRef={bufferRef} seconds={local.plotSeconds} scale={eegScale} /></div>
          <div className="signal-footer"><span><Radio size={14} /> {eeg.running ? `${Number(eeg.samples || 0).toLocaleString('pt-BR')} amostras` : 'Sinal ainda não iniciado'}</span><span className="signal-options"><span>Janela de {local.plotSeconds}s</span><label className="scale-select">Escala<select value={eegScale} onChange={(e) => setEegScale(Number(e.target.value))} aria-label="Escala vertical em microvolts"><option value={50}>±50 μV</option><option value={100}>±100 μV</option><option value={250}>±250 μV</option></select></label></span></div>
        </section>

        <aside className="insight-rail">
          <section className="surface inference-panel">
            <PanelHeading icon={BrainCircuit} title="Interpretação" action={<StatusChip tone={model ? 'positive' : 'neutral'}>{model ? 'Modelo ativo' : 'Sem modelo'}</StatusChip>} />
            <div className={`prediction ${result ? 'has-result' : ''}`}>
              {result ? <><span className="prediction-direction">{result.side === 'ESQUERDA' ? 'Esquerda' : 'Direita'} <ArrowRight className={result.side === 'ESQUERDA' ? 'flip' : ''} /></span><span className="prediction-confidence">{Math.round(result.conf * 100)}% <small>confiança</small></span></> : <><span className="waiting-mark"><Waves /></span><strong>Aguardando sinal</strong></>}
            </div>
            <div className="probability-row"><b>Esq {result ? `${Math.round(result.left * 100)}%` : '—'}</b><span className="probability-track"><i style={{ width: result ? `${Math.round(result.left * 100)}%` : '50%' }} /></span><b>{result ? `${Math.round(result.right * 100)}%` : '—'} Dir</b></div>
            {model && <div className="model-note" title={model.name}>{model.name}</div>}
            {task === 'jogo' && <div className="score-row"><span><small>Placar da sessão</small><b>Esq {placar.left} <i /> Dir {placar.right}</b></span><span><small>Acurácia VR</small><b className={accRate >= 80 ? 'positive-text' : ''}>{accRate.toFixed(1)}%</b></span></div>}
          </section>

          <section className="surface feedback-panel">
            <div className="feedback-heading"><div><h2>A IA acertou?</h2><p>{task === 'jogo' ? 'A resposta chega do VR sozinha; os botões valem como reforço manual.' : 'Confirme o resultado para ajustar o aprendizado.'}</p></div><StatusChip>{rl.enabled ? `${rlLabeled} feedbacks` : 'RL desligado'}</StatusChip></div>
            <div className="feedback-actions"><button className="confirm" onClick={() => sendFeedback(true)} disabled={!result}><Check size={18} /> Sim, acertou</button><button className="reject" onClick={() => sendFeedback(false)} disabled={!result}><X size={18} /> Não, errou</button><button className="icon-button" onClick={restoreBase} title="Restaurar modelo base" aria-label="Restaurar modelo base"><RotateCcw size={17} /></button></div>
            {rl.enabled && <p className="rl-detail">{rl.updates_applied} atualizações aplicadas</p>}
          </section>

          <section className="surface markers-panel">
            <PanelHeading icon={Radio} title="Pistas da tarefa" action={<span className="marker-count">Esq {markers.t1_count} <i /> Dir {markers.t2_count}</span>}>Envia a pista ao VR e marca o treino.</PanelHeading>
            <div className="marker-actions"><button onClick={() => sendMarker('T1')}><ArrowDownLeft size={18} /><span><b>Esquerda</b></span></button><button onClick={() => sendMarker('T2')}><ArrowUpRight size={18} /><span><b>Direita</b></span></button></div>
          </section>
        </aside>
      </div>
    </div>
  );
}
