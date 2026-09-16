/* Cliente HTTP + WebSocket do backend BrainBridge (Fase 1).
 * Base: mesma origem em dev/preview (proxy Vite). Dentro do Tauri
 * (producao), o backend sidecar sobe em 127.0.0.1:8000.
 * Contrato: { ok: true, data } | HTTP error com { detail }.
 */

const TAURI_API =
  typeof window !== 'undefined' && window.__TAURI_INTERNALS__
    ? 'http://127.0.0.1:8000'
    : '';

async function request(path, options = {}) {
  const response = await fetch(`${TAURI_API}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  let body = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  if (!response.ok) {
    const detail = body && body.detail ? body.detail : `HTTP ${response.status}`;
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return body && 'data' in body ? body.data : body;
}

const get = (path) => request(path);
const post = (path, payload) =>
  request(path, { method: 'POST', body: payload === undefined ? undefined : JSON.stringify(payload ?? {}) });
const put = (path, payload) => request(path, { method: 'PUT', body: JSON.stringify(payload ?? {}) });
const patch = (path, payload) => request(path, { method: 'PATCH', body: JSON.stringify(payload ?? {}) });

export const api = {
  health: () => get('/api/health'),
  // pacientes
  patients: () => get('/api/patients'),
  createPatient: (data) => post('/api/patients', data),
  updateHand: (id, affected_hand) => patch(`/api/patients/${id}/affected-hand`, { affected_hand }),
  recordings: (patientId) => get(`/api/patients/${patientId}/recordings`),
  calibration: (patientId) => get(`/api/patients/${patientId}/calibration`),
  // gravacao / sessao
  recordingStart: (data) => post('/api/recordings/start', data),
  recordingStop: (id, duration_seconds = 0) => post(`/api/recordings/${id}/stop`, { duration_seconds }),
  sessionStart: (data) => post('/api/sessions/start', data),
  sessionCurrent: () => get('/api/sessions/current'),
  sessionEnd: () => post('/api/sessions/end'),
  // marcadores
  markerState: () => get('/api/markers/state'),
  marker: (marker_type, task_type) => post('/api/markers', { marker_type, task_type }),
  markersReset: () => post('/api/markers/reset'),
  baselineStart: (duration_seconds = 300) => post('/api/baseline/start', { duration_seconds }),
  baselineTick: () => post('/api/baseline/tick'),
  // modelos / inferencia
  models: () => get('/api/models'),
  loadedModel: () => get('/api/models/loaded'),
  loadModel: (path) => post('/api/models/load', { path }),
  loadLatestModel: () => post('/api/models/load-latest'),
  predict: (window, input_fs) => post('/api/inference/predict', { window, input_fs }),
  eaStatus: () => get('/api/inference/ea'),
  // RL
  rlStatus: () => get('/api/rl/status'),
  rlUpdate: (payload) => post('/api/rl/update', payload),
  rlSnapshot: () => post('/api/rl/snapshot'),
  rlRestore: () => post('/api/rl/restore'),
  // dispositivos
  devices: () => get('/api/devices/status'),
  eegConnect: (opts = {}) => post('/api/devices/eeg/connect', opts),
  eegDisconnect: () => post('/api/devices/eeg/disconnect'),
  unityStart: () => post('/api/devices/unity/start'),
  unityStop: () => post('/api/devices/unity/stop'),
  unityAction: (direction) => post('/api/devices/unity/action', { direction }),
  unityPublishSession: (patient_id, task_type) => post('/api/devices/unity/publish-session', { patient_id, task_type }),
  unityVerdicts: (after = 0) => get(`/api/devices/unity/verdicts?after=${Number(after) || 0}`),
  unitySession: (data) => post('/api/devices/unity/session', data),
  unityTrigger: () => post('/api/devices/unity/trigger'),
  unityEndTask: () => post('/api/devices/unity/end-task'),
  unityEndSession: (message = '') => post('/api/devices/unity/end-session', { message }),
  esp32Connect: (port) => post('/api/devices/esp32/connect', port ? { port } : {}),
  esp32Ports: () => get('/api/devices/esp32/ports'),
  esp32Disconnect: () => post('/api/devices/esp32/disconnect'),
  esp32Action: (direction) => post('/api/devices/esp32/action', { direction }),
  // treino
  trainingStart: (csv_file_path, patient_id, auto_load = true) =>
    post('/api/training', { csv_file_path, patient_id, auto_load }),
  trainingStatus: (jobId) => get(`/api/training/${jobId}`),
  trainingCheck: (csv_file_path, required) =>
    post('/api/training/check', { csv_file_path, required }),
  // config
  config: () => get('/api/config'),
  updateConfig: (values) => put('/api/config', { values }),
};

export function wsUrl(path) {
  if (TAURI_API) return `ws://127.0.0.1:8000${path}`;
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}${path}`;
}
