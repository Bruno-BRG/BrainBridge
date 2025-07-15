import socket
import json
import threading
import time
from typing import Callable, Any, Optional

class UDPReceiver:
    """Recebe dados UDP e chama um callback."""
    def __init__(self, host: str = 'localhost', port: int = 12345):
        self.host = host
        self.port = port
        self.socket = None
        self.thread = None
        self.is_running = False
        self.callback: Optional[Callable[[Any], None]] = None

    def set_callback(self, cb: Callable[[Any], None]):
        self.callback = cb

    def start(self):
        if self.is_running:
            return
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.host, self.port))
        self.socket.settimeout(1.0)
        self.is_running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.socket:
            self.socket.close()
            self.socket = None

    def _receive_loop(self):
        while self.is_running:
            try:
                data, addr = self.socket.recvfrom(4096)
                text = data.decode('utf-8')
                try:
                    decoded = json.loads(text)
                except json.JSONDecodeError:
                    decoded = text
                if self.callback:
                    self.callback(decoded)
            except socket.timeout:
                continue
            except Exception:
                break
