import React from 'react';
import { AlertCircle, CheckCircle2, Info } from 'lucide-react';

export function Notice({ children, type = 'info' }) {
  if (!children) return null;
  const Icon = type === 'error' ? AlertCircle : type === 'success' ? CheckCircle2 : Info;
  return <div className={`notice ${type}`} role={type === 'error' ? 'alert' : 'status'}><Icon size={17} aria-hidden="true" /><span>{children}</span></div>;
}

export function EmptyState({ icon: Icon, title, children, action }) {
  return <div className="empty-state">{Icon && <Icon size={30} strokeWidth={1.5} aria-hidden="true" />}<h3>{title}</h3><p>{children}</p>{action}</div>;
}

export function PanelHeading({ icon: Icon, title, children, action }) {
  return <div className="panel-heading"><div className="panel-title">{Icon && <Icon size={18} aria-hidden="true" />}<div><h2>{title}</h2>{children && <p>{children}</p>}</div></div>{action}</div>;
}

export function StatusChip({ children, tone = '', dot = false }) {
  return <span className={`status-chip ${tone}`}>{dot && <span className="status-dot" aria-hidden="true" />}{children}</span>;
}

const LOCAL_KEY = 'brainbridge.localSettings';
export const DEFAULT_LOCAL = { eegHost: 'localhost', eegPort: 12345, eegSimulate: false, eegStream: 'raw', plotSeconds: 8, eegScale: 100, esp32Port: '' };
export function loadLocalSettings() {
  try {
    const raw = localStorage.getItem(LOCAL_KEY);
    if (!raw) return { ...DEFAULT_LOCAL };
    const parsed = JSON.parse(raw);
    return { ...DEFAULT_LOCAL, ...parsed };
  } catch {
    return { ...DEFAULT_LOCAL };
  }
}
export function saveLocalSettings(values) {
  const next = { ...loadLocalSettings(), ...values };
  try {
    localStorage.setItem(LOCAL_KEY, JSON.stringify(next));
  } catch {
    /* armazenamento indisponível: mantém em memória */
  }
  window.dispatchEvent(new Event('bb:local-settings'));
  return next;
}
