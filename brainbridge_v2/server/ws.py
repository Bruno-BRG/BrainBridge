"""WebSocket /ws/eeg: comandos + streaming do EEG em lotes de ~40ms."""

import asyncio
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from brainbridge_v2.server.eeg_hub import get_hub

router = APIRouter()


@router.websocket("/ws/eeg")
async def eeg_socket(websocket: WebSocket):
    await websocket.accept()
    hub = get_hub()
    queue = hub.subscribe()
    try:
        await websocket.send_json({"type": "status", "eeg": hub.status()})
        while True:
            try:
                incoming = await asyncio.wait_for(websocket.receive_json(), timeout=0.04)
            except asyncio.TimeoutError:
                incoming = None
            if incoming is not None:
                reply = _handle_command(incoming)
                if reply is not None:
                    await websocket.send_json(reply)
            batch, first_t = [], None
            while not queue.empty():
                try:
                    message = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if first_t is None:
                    first_t = message.get("t")
                batch.append(message.get("data"))
                if len(batch) >= 32:
                    break
            if batch:
                await websocket.send_json({"type": "eeg", "t": first_t,
                                           "count": len(batch), "batch": batch})
    except WebSocketDisconnect:
        pass
    finally:
        hub.unsubscribe(queue)


def _handle_command(message: dict):
    hub = get_hub()
    if not isinstance(message, dict):
        return {"type": "error", "detail": "Comando invalido."}
    cmd = str(message.get("cmd", "")).strip().lower()
    if cmd == "ping":
        return {"type": "pong", "t": time.time()}
    if cmd == "connect":
        try:
            status = hub.connect(
                str(message.get("host", "localhost") or "localhost"),
                int(message.get("port", 12345)),
                simulate=bool(message.get("simulate", False)),
                stream=str(message.get("stream", "raw") or "raw"))
        except Exception as exc:
            return {"type": "error", "detail": str(exc)}
        return {"type": "status", "eeg": status}
    if cmd == "disconnect":
        return {"type": "status", "eeg": hub.disconnect()}
    if cmd == "status":
        return {"type": "status", "eeg": hub.status()}
    return {"type": "error", "detail": f"Comando desconhecido: {cmd}"}
