import React, { useCallback, useEffect, useState } from 'react';
import { ArrowRight, CalendarDays, ClipboardList, FileText, Hand, Plus, RefreshCw, Search, UserRound, UsersRound } from 'lucide-react';
import { api } from '../api.js';
import { EmptyState, Notice, PanelHeading, StatusChip } from './ui.jsx';

const EMPTY_FORM = {
  name: '', age: '', sex: '', affected_hand: 'left', time_since_event: '', notes: '',
};

export default function Patients() {
  const [list, setList] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [error, setError] = useState('');
  const [form, setForm] = useState(EMPTY_FORM);
  const [saving, setSaving] = useState(false);
  const [hand, setHand] = useState('left');
  const [handSaving, setHandSaving] = useState(false);
  const [query, setQuery] = useState('');

  const fail = (err) => setError(err && err.message ? err.message : String(err));

  const loadList = useCallback(async () => {
    try {
      setList((await api.patients()) || []);
      setError('');
    } catch (err) {
      fail(err);
    }
  }, []);

  useEffect(() => {
    loadList();
  }, [loadList]);

  const select = async (patient) => {
    setSelectedId(patient.id);
    setSessions([]);
    setError('');
    if (patient.affected_hand) setHand(patient.affected_hand);
    try {
      setSessions((await api.recordings(patient.id)) || []);
    } catch (err) {
      fail(err);
    }
  };

  const selected = list.find((p) => p.id === selectedId) || null;
  const normalizedQuery = query.trim().toLocaleLowerCase('pt-BR');
  const filteredList = normalizedQuery
    ? list.filter((p) => String(p.name || '').toLocaleLowerCase('pt-BR').includes(normalizedQuery) || String(p.id).includes(normalizedQuery))
    : list;
  const set = (key) => (e) => setForm((f) => ({ ...f, [key]: e.target.value }));

  const submit = async (e) => {
    e.preventDefault();
    setError('');
    if (!form.name.trim()) {
      setError('Informe o nome do paciente.');
      return;
    }
    setSaving(true);
    try {
      await api.createPatient({
        name: form.name.trim(),
        age: Number(form.age) || 0,
        sex: form.sex,
        affected_hand: form.affected_hand || null,
        time_since_event: Number(form.time_since_event) || 0,
        notes: form.notes,
      });
      setForm(EMPTY_FORM);
      await loadList();
    } catch (err) {
      fail(err);
    } finally {
      setSaving(false);
    }
  };

  const saveHand = async () => {
    if (selectedId == null) return;
    setError('');
    setHandSaving(true);
    try {
      await api.updateHand(selectedId, hand);
      await loadList();
    } catch (err) {
      fail(err);
    } finally {
      setHandSaving(false);
    }
  };

  return (
    <div className="patients-layout">
      <Notice type="error">{error}</Notice>
      <section className="patient-directory surface">
        <PanelHeading icon={UsersRound} title="Pessoas cadastradas" action={<button className="icon-button" onClick={loadList} title="Atualizar lista" aria-label="Atualizar lista"><RefreshCw size={17} /></button>}>Selecione uma pessoa para ver detalhes e sessões.</PanelHeading>
        <div className="search-shell"><Search size={17} aria-hidden="true" /><input aria-label="Buscar paciente" placeholder="Buscar por nome ou ID" value={query} onChange={(e) => setQuery(e.target.value)} /></div>
        <div className="patient-list">
          {list.length === 0 && <EmptyState icon={UsersRound} title="Nenhum paciente ainda">Cadastre a primeira pessoa no formulário ao lado.</EmptyState>}
          {filteredList.map((p) => (
            <button key={p.id} className={`patient-row ${p.id === selectedId ? 'selected' : ''}`} onClick={() => select(p)}>
              <span className="avatar">{String(p.name || '?').trim().slice(0, 2).toUpperCase()}</span>
              <span className="patient-row-copy"><strong>{p.name}</strong><small>ID {p.id} · {p.age ? `${p.age} anos` : 'idade não informada'}</small></span>
              <ArrowRight size={17} aria-hidden="true" />
            </button>
          ))}
          {list.length > 0 && filteredList.length === 0 && <EmptyState icon={Search} title="Nenhum resultado">Tente outro nome ou ID.</EmptyState>}
        </div>
      </section>

      <div className="patient-main">
        <section className="patient-detail surface">
          {selected ? <>
            <PanelHeading icon={UserRound} title={selected.name} action={<StatusChip tone="positive">Paciente selecionado</StatusChip>}>ID {selected.id} · cadastro clínico</PanelHeading>
            <div className="patient-facts">
              <div><small>Idade</small><strong>{selected.age || '—'}{selected.age ? ' anos' : ''}</strong></div>
              <div><small>Sexo</small><strong>{selected.sex || '—'}</strong></div>
              <div><small>Tempo desde o evento</small><strong>{selected.time_since_event || 0} meses</strong></div>
            </div>
            <div className="hand-setting">
              <span><Hand size={18} /><span><small>Lateralidade afetada</small><strong>{selected.affected_hand === 'right' ? 'Direita' : selected.affected_hand === 'left' ? 'Esquerda' : 'Não informada'}</strong></span></span>
              <div><select aria-label="Mão afetada" value={hand} onChange={(e) => setHand(e.target.value)}><option value="left">Esquerda</option><option value="right">Direita</option></select><button className="compact-button primary" onClick={saveHand} disabled={handSaving}>{handSaving ? 'Salvando…' : 'Atualizar'}</button></div>
            </div>
            {selected.notes && <div className="clinical-notes"><FileText size={17} /><span><small>Observações clínicas</small><p>{selected.notes}</p></span></div>}
            <div className="sessions-heading"><div><ClipboardList size={18} /><h3>Sessões registradas</h3></div><StatusChip>{sessions.length} {sessions.length === 1 ? 'sessão' : 'sessões'}</StatusChip></div>
            <div className="session-list">
              {sessions.length === 0 && <EmptyState icon={CalendarDays} title="Nenhuma sessão registrada">As gravações vinculadas a esta pessoa aparecerão aqui.</EmptyState>}
              {sessions.map((s) => <div key={s.id} className="session-row"><span className="session-symbol"><ActivityIcon /></span><span><strong>{s.task_type || 'Sessão'}</strong><small>{s.filename}</small></span><span className="session-meta"><b>{s.duration != null ? `${s.duration}s` : '—'}</b><small>{s.start_time || 'Data não informada'}</small></span></div>)}
            </div>
          </> : <EmptyState icon={UserRound} title="Selecione um paciente">Escolha uma pessoa na lista para consultar dados clínicos e sessões.</EmptyState>}
        </section>

        <section className="new-patient surface">
          <PanelHeading icon={Plus} title="Cadastrar paciente">Crie um registro para vincular futuras sessões.</PanelHeading>
          <form className="form-grid" onSubmit={submit}>
            <label className="field span-2">Nome completo<input value={form.name} onChange={set('name')} placeholder="Ex.: Ana Silva" /></label>
            <label className="field">Idade<input type="number" min="0" value={form.age} onChange={set('age')} placeholder="Ex.: 54" /></label>
            <label className="field">Sexo<input value={form.sex} onChange={set('sex')} placeholder="Ex.: F" /></label>
            <label className="field">Mão afetada<select value={form.affected_hand} onChange={set('affected_hand')}><option value="left">Esquerda</option><option value="right">Direita</option></select></label>
            <label className="field">Tempo desde o evento<input type="number" min="0" value={form.time_since_event} onChange={set('time_since_event')} placeholder="Meses" /></label>
            <label className="field span-2">Observações<textarea value={form.notes} onChange={set('notes')} placeholder="Contexto clínico relevante para a sessão…" rows={3} /></label>
            <div className="form-actions span-2"><span>Os dados ficam no ambiente local.</span><button className="primary-button" type="submit" disabled={saving}><Plus size={17} />{saving ? 'Cadastrando…' : 'Cadastrar paciente'}</button></div>
          </form>
        </section>
      </div>
    </div>
  );
}

function ActivityIcon() {
  return <span className="activity-glyph" aria-hidden="true"><i /><i /><i /><i /></span>;
}
