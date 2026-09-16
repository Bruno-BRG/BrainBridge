import React, { useEffect, useState } from 'react';
import { Activity, BrainCircuit, GraduationCap, Settings2, UsersRound, Wifi, WifiOff, ArrowUpRight } from 'lucide-react';
import { api } from './api.js';
import Streaming from './components/Streaming.jsx';
import Patients from './components/Patients.jsx';
import Training from './components/Training.jsx';
import Settings from './components/Settings.jsx';

const ROUTES = {
  stream: { label: 'Sessão ao vivo', title: 'Sua próxima conexão.', description: 'Do sinal à intenção. Prepare e acompanhe sua sessão.', icon: Activity },
  patients: { label: 'Pacientes', title: 'Cada pessoa, uma história.', description: 'Cadastros e gravações para acompanhar cada sessão.', icon: UsersRound },
  training: { label: 'Treino e modelos', title: 'Aprender. Calibrar. Evoluir.', description: 'Transforme suas gravações em modelos personalizados.', icon: GraduationCap },
  settings: { label: 'Ajustes', title: 'Precisão em cada detalhe.', description: 'Configure a calibração e os limites do aprendizado online.', icon: Settings2 },
};

export default function App() {
  const [tab, setTab] = useState('stream');
  const [visited, setVisited] = useState({ stream: true });
  const [health, setHealth] = useState(null);
  const navigate = (next) => { setVisited((old) => ({ ...old, [next]: true })); setTab(next); };

  useEffect(() => {
    let alive = true;
    const check = () => api.health().then(() => alive && setHealth(true)).catch(() => alive && setHealth(false));
    check();
    const timer = setInterval(check, 10000);
    return () => { alive = false; clearInterval(timer); };
  }, []);

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">Ir para o conteúdo</a>
      <aside className="app-nav" aria-label="Navegação principal">
        <div className="brand-lockup">
          <span className="brand-mark"><BrainCircuit aria-hidden="true" /></span>
          <span className="brand-copy"><strong>BrainBridge<span>.</span></strong><small>Intenção em movimento</small></span>
        </div>
        <nav className="nav-list">
          {Object.entries(ROUTES).map(([key, route]) => {
            const Icon = route.icon;
            return <button key={key} className={`nav-item ${tab === key ? 'active' : ''}`} onClick={() => navigate(key)} aria-current={tab === key ? 'page' : undefined}><Icon aria-hidden="true" /><span>{route.label}</span>{tab === key && <span className="nav-indicator" />}</button>;
          })}
        </nav>
        <div className="nav-note"><span className="bridge-lines" aria-hidden="true"><i /><i /><i /><i /><i /><i /><i /></span><p>Uma ponte entre<br /><strong>intenção e movimento.</strong></p></div>
        <div className="nav-footer">
          <div className={`system-status ${health ? 'online' : ''}`}>
            {health ? <Wifi aria-hidden="true" /> : <WifiOff aria-hidden="true" />}
            <span><strong>{health === null ? 'Conectando…' : health ? 'Sistema online' : 'Sistema offline'}</strong><small>{health === false ? 'Verifique o serviço local' : 'Ambiente de trabalho local'}</small></span>
          </div>
          <div className="version-row"><span>BrainBridge</span><span className="version-label">v2.0</span></div>
        </div>
      </aside>
      <main className="app-main" id="main-content" tabIndex={-1}>
        <div className="workspace-bar"><span>Workspace <span className="breadcrumb-slash">/</span> <strong>{ROUTES[tab].label}</strong></span><span className="local-label"><span className="status-dot" /> Ambiente local</span></div>
        <header className="workspace-heading"><div><h1>{ROUTES[tab].title}</h1><p>{ROUTES[tab].description}</p></div>{tab === 'stream' && <button className="text-button heading-link" onClick={() => navigate('patients')}>Gerenciar pacientes <ArrowUpRight size={16} /></button>}</header>
        <div className="page-content">
          <div hidden={tab !== 'stream'}><Streaming active={tab === 'stream'} onNavigate={navigate} /></div>
          {visited.patients && <div hidden={tab !== 'patients'}><Patients active={tab === 'patients'} /></div>}
          {visited.training && <div hidden={tab !== 'training'}><Training active={tab === 'training'} /></div>}
          {visited.settings && <div hidden={tab !== 'settings'}><Settings active={tab === 'settings'} /></div>}
        </div>
        <footer className="workspace-footer"><span>BrainBridge · Interface cérebro–computador</span><span>Dados e dispositivos no seu ambiente</span></footer>
      </main>
    </div>
  );
}
