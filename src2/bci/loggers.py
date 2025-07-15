import csv
import os
import threading
import time
from datetime import datetime
from typing import Any, List, Optional, Dict
import logging

from bci.network import UDPReceiver

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class CSVDataLogger:
    """Logger genérico para salvar pacotes UDP em CSV."""
    def __init__(self, csv_filename: Optional[str] = None, host='localhost', port=12345):
        self.host = host; self.port = port
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = csv_filename or f"udp_data_{timestamp}.csv"
        self.csv_filepath = os.path.join(os.getcwd(), self.csv_filename)

        self.udp = UDPReceiver(host, port)
        self.is_logging = False
        self.buffer: List[Dict[str, Any]] = []
        self.buffer_lock = threading.Lock()
        self.flush_size = 50
        self.flush_interval = 2.0
        self.last_flush = time.time()

        self.headers = ['timestamp','source_ip','source_port','data_type','raw_data']
        self.headers_written = False

    def _flush(self):
        with self.buffer_lock:
            if not self.buffer: return
            data, self.buffer = self.buffer, []
        exist = os.path.exists(self.csv_filepath)
        with open(self.csv_filepath, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=self.headers)
            if not exist or not self.headers_written:
                w.writeheader(); self.headers_written=True
            w.writerows(data)
        logger.info(f"Salvos {len(data)} registros em {self.csv_filename}")
        self.last_flush = time.time()

    def _check_flush(self):
        if len(self.buffer)>=self.flush_size or (time.time()-self.last_flush)>=self.flush_interval:
            threading.Thread(target=self._flush, daemon=True).start()

    def _on_data(self, packet):
        if not self.is_logging: return
        recs = self.udp.get_latest_data(1)
        if recs:
            r = recs[0]
            row = {
                'timestamp': datetime.fromtimestamp(r['timestamp']).isoformat(),
                'source_ip': r['address'][0],
                'source_port': r['address'][1],
                'data_type': type(r['data']).__name__,
                'raw_data': str(r['data'])
            }
            with self.buffer_lock:
                self.buffer.append(row)
            self._check_flush()

    def start(self):
        if self.is_logging: return
        self.udp.set_callback(self._on_data)
        self.udp.start()
        self.is_logging = True
        logger.info("Iniciou logging UDP")

    def stop(self):
        if not self.is_logging: return
        self.is_logging = False
        self.udp.stop()
        self._flush()

# Aqui você pode importar/mover o OpenBCICSVLogger e o SimpleCSVLogger do seu arquivo original,
# corrigindo duplicações de import e garantindo que todos usem CSVDataLogger como fallback.
# Por brevidade, não repito aqui, mas basta mover as classes e ajustar imports.
