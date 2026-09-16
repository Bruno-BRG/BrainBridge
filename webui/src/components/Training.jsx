import React, { useEffect, useRef, useState } from 'react';
import { BrainCircuit, Check, CheckCircle2, ChevronRight, Database, FileCheck2, FolderOpen, GraduationCap, LoaderCircle, Play, RefreshCw, Rocket, Sparkles } from 'lucide-react';
import { api } from '../api.js';
import { EmptyState, Notice, PanelHeading, StatusChip } from './ui.jsx';

function parseMetrics(messages) {
  let valAccuracy = null;
  let delta = null;
  for (const line of messages || []) {
    const acc = String(line).match(/(\d+(?:[.,]\d+)?)\s*%/);
    if (/val/i.test(line) && acc && valAccuracy === null) {
      valAccuracy = acc[0];
    }
    const d = String(line).match(/delta[^:]*:\s*(.+)/i);
    if (d) delta = d[1].trim();
  }
  return { valAccuracy, delta };
}

export default function Training() {
  const [csv, setCsv] = useState('');
  const [patientId, setPatientId] = useState('');
  const [check, setCheck] = useState(null);
  const [checking, setChecking] = useState(false);
  const [jobId, setJobId] = useState(null);
  const [job, setJob] = useState(null);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState('');
  const [models, setModels] = useState([]);
  const [loaded, setLoaded] = useState(null);
  const [loadingModel, setLoadingModel] = useState(false);
  const [patients, setPatients] = useState([]);
  const timerRef = useRef(null);

  const fail = (err) => setError(err && err.message ? err.message : String(err));

  const refreshModels = async () => {
    try {
      const [list, current, patientList] = await Promise.all([api.models(), api.loadedModel(), api.patients()]);
      setModels(list || []);
      setLoaded(current || null);
      setPatients(patientList || []);
    } catch (err) {
      fail(err);
    }
  };

  useEffect(() => {
    refreshModels();
    return () => clearInterval(timerRef.current);
  }, []);

  useEffect(() => {
    clearInterval(timerRef.current);
    if (!jobId) return undefined;
    const poll = async () => {
      try {
        const st = await api.trainingStatus(jobId);
        setJob(st);
        if (st.status === 'done' || st.status === 'error') {
          clearInterval(timerRef.current);
          refreshModels();
        }
      } catch (err) {
        fail(err);
        clearInterval(timerRef.current);
      }
    };
    poll();
    timerRef.current = setInterval(poll, 2000);
    return () => clearInterval(timerRef.current);
  }, [jobId]);

  const verify = async () => {
    setError('');
    setCheck(null);
    if (!csv.trim()) {
      setError('Informe o caminho do CSV.');
      return;
    }
    setChecking(true);
    try {
      setCheck(await api.trainingCheck(csv.trim()));
    } catch (err) {
      fail(err);
    } finally {
      setChecking(false);
    }
  };

  const start = async () => {
    setError('');
    if (!csv.trim() || !patientId) {
      setError('Informe o CSV e o ID do paciente.');
      return;
    }
    setStarting(true);
    try {
      const out = await api.trainingStart(csv.trim(), Number(patientId), true);
      setJobId(out.job_id);
      setJob(null);
    } catch (err) {
      fail(err);
    } finally {
      setStarting(false);
    }
  };

  const loadOne = async (path) => {
    setError('');
    setLoadingModel(true);
    try {
      await api.loadModel(path);
      await refreshModels();
    } catch (err) {
      fail(err);
    } finally {
      setLoadingModel(false);
    }
  };

  const loadLatest = async () => {
    setError('');
    setLoadingModel(true);
    try {
      await api.loadLatestModel();
      await refreshModels();
    } catch (err) {
      fail(err);
    } finally {
      setLoadingModel(false);
    }
  };

  const checkFailed = check && !check.ok;
  const messages = (job && job.messages) || [];
  const { valAccuracy, delta } = parseMetrics(messages);
  const resultAcc = job && job.result && job.result.val_accuracy != null
    ? `${(Number(job.result.val_accuracy) * 100).toFixed(1)}%`
    : valAccuracy;
  const training = job && job.status !== 'done' && job.status !== 'error';

  return (
    <div className="training-layout">
      <Notice type="error">{error}</Notice>
      <section className="training-flow surface">
        <PanelHeading icon={GraduationCap} title="Nova calibração" action={training ? <StatusChip tone="progress"><LoaderCircle className="spin" size={14} /> Treinando</StatusChip> : null}>Converta uma sessão gravada em um modelo específico para o paciente.</PanelHeading>
        <div className="training-steps">
          <div className={csv ? 'complete' : 'current'}><i>{csv ? <Check size={13} /> : '1'}</i><span><b>Escolher gravação</b><small>Arquivo CSV de treino</small></span></div><ChevronRight />
          <div className={check ? check.ok ? 'complete' : 'current' : ''}><i>{check && check.ok ? <Check size={13} /> : '2'}</i><span><b>Validar trials</b><small>Integridade da sessão</small></span></div><ChevronRight />
          <div className={job && job.status === 'done' ? 'complete' : training ? 'current' : ''}><i>{job && job.status === 'done' ? <Check size={13} /> : '3'}</i><span><b>Treinar</b><small>Calibrar e carregar</small></span></div>
        </div>

        <div className="training-form">
          <label className="field span-2">Arquivo da gravação<div className="input-with-icon"><FolderOpen size={17} /><input value={csv} onChange={(e) => { setCsv(e.target.value); setCheck(null); }} placeholder="Cole o caminho do CSV gerado na sessão de treino" /></div></label>
          <label className="field">Paciente<select value={patientId} onChange={(e) => setPatientId(e.target.value)}><option value="">Selecione o paciente</option>{patients.map((p) => <option key={p.id} value={p.id}>{p.name} · ID {p.id}</option>)}</select></label>
          <div className="validation-action"><button className="secondary-button" onClick={verify} disabled={checking || !csv.trim()}>{checking ? <LoaderCircle className="spin" size={17} /> : <FileCheck2 size={17} />}{checking ? 'Verificando…' : 'Verificar gravação'}</button></div>
        </div>

        {check && <div className={`trial-result ${check.ok ? 'valid' : 'invalid'}`}><span>{check.ok ? <CheckCircle2 /> : <FileCheck2 />}<span><strong>{check.trials} trials identificados</strong><small>Mínimo recomendado: {check.required} · {check.ok ? 'gravação pronta para treino' : 'amostra abaixo do mínimo'}</small></span></span>{check.ok && <StatusChip tone="positive">Validado</StatusChip>}</div>}

        <div className="train-launch"><div><Sparkles size={20} /><span><strong>Fine-tuning personalizado</strong><small>O modelo concluído será carregado automaticamente.</small></span></div><button className="primary-button" onClick={start} disabled={starting || training || !!checkFailed || !csv.trim() || !patientId}>{starting || training ? <LoaderCircle className="spin" size={18} /> : <Rocket size={18} />}{training ? 'Treinando…' : starting ? 'Iniciando…' : 'Iniciar treinamento'}</button></div>
        {checkFailed && <button className="text-button danger-text" onClick={start} disabled={starting || training}>Treinar assim mesmo com {check.trials} trials</button>}

        {job && <div className="training-progress"><div className="progress-heading"><span><BrainCircuit size={18} /><strong>Execução {jobId}</strong></span><StatusChip tone={job.status === 'done' ? 'positive' : job.status === 'error' ? 'negative' : 'progress'}>{job.status}</StatusChip></div><div className="training-log">{messages.length ? messages.join('\n') : 'Preparando ambiente de treino…'}</div>{job.status === 'done' && <Notice type="success">Treino concluído{resultAcc ? ` · validação ${resultAcc}` : ''}{delta ? ` · ${delta}` : ''}</Notice>}{job.status === 'error' && <Notice type="error">{job.error || 'Falha no treino.'}</Notice>}</div>}
      </section>

      <section className="model-library surface">
        <PanelHeading icon={Database} title="Biblioteca de modelos" action={<button className="icon-button" onClick={refreshModels} title="Atualizar modelos" aria-label="Atualizar modelos"><RefreshCw size={17} /></button>}>Escolha qual modelo interpreta o sinal ao vivo.</PanelHeading>
        <div className={`active-model ${loaded ? '' : 'empty'}`}><div className="model-orbit"><BrainCircuit /></div><div><small>Modelo ativo</small><strong>{loaded ? loaded.name : 'Nenhum modelo carregado'}</strong><p>{loaded ? 'Pronto para inferência na sessão ao vivo.' : 'Carregue um modelo disponível abaixo.'}</p></div>{loaded && <StatusChip tone="positive" dot>Em uso</StatusChip>}</div>
        <div className="library-toolbar"><span>{models.length} {models.length === 1 ? 'modelo disponível' : 'modelos disponíveis'}</span><button className="secondary-button compact" onClick={loadLatest} disabled={loadingModel}><Play size={15} /> Carregar mais recente</button></div>
        <div className="model-list">
          {models.length === 0 && <EmptyState icon={Database} title="Biblioteca vazia">Treine um modelo para que ele apareça aqui.</EmptyState>}
          {models.map((m, index) => { const active = loaded && loaded.path === m.path; return <div key={m.path} className={`model-row ${active ? 'selected' : ''}`} title={m.path}><span className="model-index">{String(index + 1).padStart(2, '0')}</span><span><strong>{m.name}</strong></span>{active ? <StatusChip tone="positive">Ativo</StatusChip> : <button className="compact-button" onClick={() => loadOne(m.path)} disabled={loadingModel}>Carregar</button>}</div>; })}
        </div>
      </section>
    </div>
  );
}
