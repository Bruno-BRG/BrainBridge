import json
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
from bci.network import UDPReceiver
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class RealTimeUDPConverter:
    """Converte pacotes UDP para o formato OpenBCI em CSV."""
    def __init__(self, csv_filename: Optional[str] = None, host='localhost', port=12345):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = csv_filename or f"openbci_converted_{timestamp}.csv"
        self.csv_filepath = os.path.join(os.getcwd(), self.csv_filename)

        self.udp = UDPReceiver(host, port)
        self.is_converting = False
        self.buffer: List[Dict] = []
        self.buffer_lock = threading.Lock()
        self.flush_size = 30
        self.flush_interval = 5.0
        self.last_flush = time.time()

        self.headers = [
            'Sample Index', *[f'EXG Channel {i}' for i in range(16)],
            'Accel0','Accel1','Accel2',
            *[f'Other{i}' for i in range(7)],
            'Analog0','Analog1','Analog2','Timestamp','Other.7','TimestampFmt','Annotations'
        ]
        self.index = 0

    # métodos de _convert_to_openbci, _process_timeseries_raw etc. idênticos ao original,
    # mas movidos para cá e usando self.buffer / self._flush.

    # ...
