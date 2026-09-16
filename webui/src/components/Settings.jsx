import React, { useCallback, useEffect, useState } from 'react';
import { AlertTriangle, BrainCircuit, Check, Cpu, FlaskConical, Monitor, RefreshCw, RotateCcw, Save, ShieldCheck, SlidersHorizontal, Sparkles } from 'lucide-react';
import { api } from '../api.js';
import { Notice, PanelHeading, StatusChip, loadLocalSettings, saveLocalSettings } from './ui.jsx';

const FIELDS = [
  { key: 'calib_trials_required', type: 'int', label: 'Trials mínimos para treinar', desc: 'Trials T1/T2 mínimos exigidos no CSV de treino.' },
  { key: 'calib_epochs', type: 'int', label: 'Épocas da calibração', desc: 'Épocas de fine-tuning no treino do paciente.' },
  { key: 'calib_lr', type: 'float', label: 'Taxa de aprendizado (calibração)', desc: 'Learning rate do treino de calibração.' },
  { key: 'calib_freeze_backbone', type: 'bool', label: 'Congelar backbone na calibração', desc: 'Se ligado, treina só a camada final.' },
  { key: 'rl_enabled', type: 'bool', label: 'RL online ativado', desc: 'Permite updates online a partir de feedbacks.' },
  { key: 'rl_batch_k', type: 'int', label: 'Feedbacks por update (K)', desc: 'Aplica um update sozinho a cada K feedbacks rotulados.' },
  { key: 'rl_epochs', type: 'int', label: 'Épocas do RL', desc: 'Épocas de cada update online.' },
  { key: 'rl_lr', type: 'float', label: 'Taxa de aprendizado (RL)', desc: 'Learning rate dos updates online.' },
  { key: 'rl_max_updates', type: 'int', label: 'Máximo de updates RL', desc: 'Trava de segurança: updates por sessão.' },
  { key: 'rl_mistake_weight', type: 'float', label: 'Peso do erro no RL', desc: 'Quanto o erro pesa mais que o acerto.' },
  { key: 'rl_buffer_max', type: 'int', label: 'Tamanho máximo do buffer RL', desc: 'Feedbacks guardados antes de descartar os antigos.' },
];

const coerce = (type, value) => {
  if (type === 'bool') return !!value;
  const n = Number(value);
  return type === 'int' ? Math.trunc(n) : n;
};

export default function Settings() {
  const [config, setConfig] = useState(null);
  const [draft, setDraft] = useState({});
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [saving, setSaving] = useState(false);
  const [rl, setRl] = useState(null);
  const [device, setDevice] = useState(loadLocalSettings);

  const setDeviceField = (key, value) => setDevice(saveLocalSettings({ [key]: value }));
  const [ports, setPorts] = useState([]);
  const [portsLoading, setPortsLoading] = useState(false);
  const loadPorts = useCallback(async () => {
    setPortsLoading(true);
    try {
      const data = await api.esp32Ports();
      setPorts(data.ports || []);
    } catch {
      setPorts([]);
    } finally {
      setPortsLoading(false);
    }
  }, []);

  useEffect(() => { loadPorts(); }, [loadPorts]);

  const fail = (err) => {
    setError(err && err.message ? err.message : String(err));
    setSuccess('');
  };

  const reload = useCallback(async () => {
    setError('');
    setSuccess('');
    try {
      const [cfg, rlStatus] = await Promise.all([api.config(), api.rlStatus()]);
      setConfig(cfg);
      setDraft(cfg);
      setRl(rlStatus);
    } catch (err) {
      fail(err);
    }
  }, []);

  useEffect(() => {
    reload();
  }, [reload]);

  const changedKeys = config
    ? FIELDS.filter((f) => config[f.key] !== undefined && coerce(f.type, draft[f.key]) !== config[f.key])
    : [];

  const save = async () => {
    setError('');
    setSuccess('');
    const values = {};
    for (const f of changedKeys) values[f.key] = coerce(f.type, draft[f.key]);
    if (Object.keys(values).length === 0) {
      setSuccess('Nada a salvar.');
      return;
    }
    setSaving(true);
    try {
      const updated = await api.updateConfig(values);
      setConfig((c) => ({ ...c, ...updated }));
      setDraft((d) => ({ ...d, ...updated }));
      setSuccess('Configuração salva.');
    } catch (err) {
      fail(err);
    } finally {
      setSaving(false);
    }
  };

  const restore = async () => {
    setError('');
    setSuccess('');
    try {
      await api.rlRestore();
      const st = await api.rlStatus();
      setRl(st);
      setSuccess('Modelo restaurado para o checkpoint pré-RL.');
    } catch (err) {
      fail(err);
    }
  };

  if (!config) return <div className="settings-loading surface"><span className="loading-line" /><span>Carregando parâmetros…</span>{error && <Notice type="error">{error}</Notice>}</div>;

  const renderField = (f) => (
    <label className={`setting-row ${f.type === 'bool' ? 'boolean' : ''}`} key={f.key}>
      <span><strong>{f.label}</strong><small>{f.desc}</small></span>
      {f.type === 'bool' ? <span className="setting-toggle"><input type="checkbox" checked={!!draft[f.key]} onChange={(e) => setDraft((d) => ({ ...d, [f.key]: e.target.checked }))} /><span className="switch" /><em>{draft[f.key] ? 'Ativado' : 'Desativado'}</em></span> : <span className="number-setting"><input type="number" step={f.type === 'int' ? '1' : 'any'} value={draft[f.key] ?? ''} onChange={(e) => setDraft((d) => ({ ...d, [f.key]: e.target.value }))} /></span>}
    </label>
  );

  return (
    <div className="settings-layout">
      <Notice type="error">{error}</Notice><Notice type="success">{success}</Notice>
      <div className="settings-main">
        <section className="settings-section surface">
          <PanelHeading icon={FlaskConical} title="Calibração do paciente" action={<StatusChip>Fine-tuning</StatusChip>}>Defina quando uma gravação está pronta e como o modelo será ajustado.</PanelHeading>
          <div className="settings-list">{FIELDS.filter((f) => f.key.startsWith('calib_') && config[f.key] !== undefined).map(renderField)}</div>
        </section>
        <section className="settings-section surface">
          <PanelHeading icon={Sparkles} title="Aprendizado online" action={<StatusChip tone={rl && rl.enabled ? 'positive' : 'neutral'} dot>{rl && rl.enabled ? 'Ativo' : 'Inativo'}</StatusChip>}>Ajuste como os feedbacks da sessão atualizam o modelo em uso.</PanelHeading>
          <div className="settings-list">{FIELDS.filter((f) => f.key.startsWith('rl_') && config[f.key] !== undefined).map(renderField)}</div>
        </section>
        <section className="settings-section surface">
          <PanelHeading icon={Cpu} title="Dispositivos" action={<StatusChip>Local</StatusChip>}>Conexão com o hardware de aquisição. Vale a partir da próxima conexão do EEG.</PanelHeading>
          <div className="settings-list">
            <label className="setting-row"><span><strong>Host do EEG</strong><small>Endereço do amplificador ou do serviço de aquisição.</small></span><span className="number-setting wide"><input type="text" value={device.eegHost} onChange={(e) => setDeviceField('eegHost', e.target.value)} /></span></label>
            <label className="setting-row"><span><strong>Porta do EEG</strong><small>Porta UDP do stream de 16 canais.</small></span><span className="number-setting"><input type="number" min="1" max="65535" value={device.eegPort} onChange={(e) => setDeviceField('eegPort', e.target.value)} /></span></label>
            <label className="setting-row"><span><strong>Stream do OpenBCI</strong><small>Bruto (timeSeriesRaw) ou filtrado (timeSeries) enviado pelo OpenBCI GUI.</small></span><span className="number-setting wide"><select value={device.eegStream === 'filtered' ? 'filtered' : 'raw'} onChange={(e) => setDeviceField('eegStream', e.target.value)}><option value="raw">Bruto · timeSeriesRaw</option><option value="filtered">Filtrado · timeSeries</option></select></span></label>
            <label className="setting-row boolean"><span><strong>Simular sinal sem hardware</strong><small>Gera EEG sintético quando não há amplificador conectado.</small></span><span className="setting-toggle"><input type="checkbox" checked={device.eegSimulate !== false} onChange={(e) => setDeviceField('eegSimulate', e.target.checked)} /><span className="switch" /><em>{device.eegSimulate !== false ? 'Ativado' : 'Desativado'}</em></span></label>
            <label className="setting-row"><span><strong>Porta da órtese</strong><small>Vazio = detectar sozinho. Ex.: /dev/ttyUSB0.</small></span><span className="number-setting wide"><input type="text" value={device.esp32Port || ''} placeholder="automático" onChange={(e) => setDeviceField('esp32Port', e.target.value)} /></span></label>
            <div className="setting-row"><span><strong>Portas seriais detectadas</strong><small>{ports.length === 0 ? 'Nenhuma porta encontrada nesta máquina.' : ports.map((p) => p.port).join(' · ')}</small></span><span><button className="compact-button" onClick={loadPorts} disabled={portsLoading}>{portsLoading ? 'Lendo…' : 'Listar'}</button></span></div>
          </div>
        </section>
        <section className="settings-section surface">
          <PanelHeading icon={Monitor} title="Exibição" action={<StatusChip>Local</StatusChip>}>Preferências visuais desta máquina. Aplicam na hora.</PanelHeading>
          <div className="settings-list">
            <label className="setting-row"><span><strong>Janela do gráfico (segundos)</strong><small>Quanto do sinal aparece no painel ao vivo, de 2 a 30.</small></span><span className="number-setting"><input type="number" min="2" max="30" value={device.plotSeconds} onChange={(e) => setDeviceField('plotSeconds', Math.min(30, Math.max(2, Number(e.target.value) || 8)))} /></span></label>
          </div>
        </section>
      </div>

      <aside className="settings-side">
        <section className="settings-summary surface">
          <PanelHeading icon={SlidersHorizontal} title="Alterações pendentes">Revise antes de aplicar ao ambiente.</PanelHeading>
          {changedKeys.length === 0 ? <div className="all-saved"><span><Check size={19} /></span><div><strong>Tudo atualizado</strong><p>Nenhuma alteração pendente.</p></div></div> : <div className="changes-list"><StatusChip tone="progress">{changedKeys.length} {changedKeys.length === 1 ? 'alteração' : 'alterações'}</StatusChip>{changedKeys.map((f) => <div key={f.key}><span>{f.label}</span><strong>{f.type === 'bool' ? (draft[f.key] ? 'Ativado' : 'Desativado') : String(draft[f.key])}</strong></div>)}</div>}
          <button className="primary-button full" onClick={save} disabled={saving || changedKeys.length === 0}><Save size={17} />{saving ? 'Salvando…' : 'Salvar alterações'}</button>
          <button className="secondary-button full" onClick={reload}><RefreshCw size={16} /> Descartar e recarregar</button>
        </section>

        <section className="safety-panel surface">
          <div className="safety-title"><ShieldCheck size={20} /><div><h2>Checkpoint de segurança</h2><p>Proteção do aprendizado online</p></div></div>
          <div className="safety-status"><span><BrainCircuit size={17} /> Atualizações aplicadas</span><strong>{rl ? rl.updates_applied : '—'}</strong></div>
          <div className="safety-status"><span><RotateCcw size={17} /> Modelo base salvo</span><strong>{rl ? (rl.has_snapshot ? 'Sim' : 'Não') : '—'}</strong></div>
          <div className="safety-warning"><AlertTriangle size={16} /><span>Restaurar remove os ajustes online desta sessão e retorna ao checkpoint pré-RL.</span></div>
          <button className="danger-outline full" onClick={restore} disabled={!rl || !rl.has_snapshot}><RotateCcw size={16} /> Restaurar modelo base</button>
        </section>
      </aside>
    </div>
  );
}
