import { useEffect, useRef, useState } from 'react';
import { wsUrl } from './api.js';

/* Hook do stream EEG via /ws/eeg.
 * Conecta sob demanda (connectOpts) e entrega lotes de amostras 16ch.
 * onBatch(batch: number[][]) chamado por flush (~40ms).
 */
export function useEegStream(connectOpts, { enabled = false, onBatch = null, onError = null } = {}) {
  const [status, setStatus] = useState({ running: false, mode: 'idle', samples: 0 });
  const [connected, setConnected] = useState(false);
  const socketRef = useRef(null);
  const optsRef = useRef(connectOpts);
  optsRef.current = connectOpts;
  const cbRef = useRef(onBatch);
  cbRef.current = onBatch;
  const errRef = useRef(onError);
  errRef.current = onError;

  useEffect(() => {
    if (!enabled) return undefined;
    let closed = false;
    const socket = new WebSocket(wsUrl('/ws/eeg'));
    socketRef.current = socket;
    socket.onopen = () => {
      setConnected(true);
      try {
        socket.send(JSON.stringify({ cmd: 'connect', ...(optsRef.current || {}) }));
      } catch {
        /* ignore */
      }
    };
    socket.onmessage = (event) => {
      let message = null;
      try {
        message = JSON.parse(event.data);
      } catch {
        return;
      }
      if (message.type === 'status' && message.eeg) setStatus(message.eeg);
      else if (message.type === 'eeg' && Array.isArray(message.batch) && cbRef.current) {
        cbRef.current(message.batch);
      }
      else if (message.type === 'error' && errRef.current) {
        errRef.current(new Error(message.detail || 'Falha na conexão do EEG.'));
      }
    };
    socket.onclose = () => {
      if (!closed) setConnected(false);
    };
    return () => {
      closed = true;
      try {
        socket.close();
      } catch {
        /* ignore */
      }
      socketRef.current = null;
    };
  }, [enabled]);

  const send = (payload) => {
    const socket = socketRef.current;
    if (socket && socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(payload));
  };

  return { status, connected, send };
}
