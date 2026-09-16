"""BrainBridge HTTP + WebSocket server (Fase 1 da migracao Tauri).

Reaproveita 100% da regra de negocio: os mesmos controllers/casos de uso
da UI PyQt, sem nenhuma dependencia Qt. O frontend web (Fase 2) e o
Tauri (Fase 3) falam com este backend via REST + WebSocket.
"""
