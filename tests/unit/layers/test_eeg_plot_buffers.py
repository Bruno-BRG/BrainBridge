import os
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtTest import QTest
from brainbridge_v2.presentation.gui.widgets.eeg_plot import _PyQtGraphBackend, _MatplotlibBackend


@pytest.mark.parametrize("backend_type", [_PyQtGraphBackend, _MatplotlibBackend])
def test_bounded_owned_samples_and_batched_render(backend_type):
    app = QApplication.instance() or QApplication([])
    parent = QWidget()
    backend = backend_type(parent)
    assert 30 <= backend.timer.interval() <= 50
    backend.timer.stop()
    backend.current_time = 86400.0
    targets = backend.curves if hasattr(backend, "curves") else backend.lines
    method = "setData" if hasattr(backend, "curves") else "set_data"
    for target in targets:
        setattr(target, method, Mock())
    sample = np.arange(16, dtype=np.float32)
    for _ in range(backend.MAX_SAMPLES + 500):
        backend.add_data(sample)
    sample[:] = -999
    assert len(backend.data_buffer) == len(backend.time_buffer) == backend.MAX_SAMPLES
    np.testing.assert_array_equal(backend.data_buffer[-1], np.arange(16))
    for target in targets:
        getattr(target, method).assert_not_called()
    backend._flush_plot()
    backend._flush_plot()
    for target in targets:
        call = getattr(target, method)
        call.assert_called_once()
        assert len(call.call_args.args[0]) <= backend.MAX_SAMPLES
        assert np.all(np.diff(call.call_args.args[0]) > 0)
    before = backend.current_time
    for invalid in (np.zeros(15), np.zeros((16, 2)), np.full(16, np.nan)):
        backend.add_data(invalid)
    assert backend.current_time == before
    backend.add_data(np.zeros(16))
    backend.timer.start()
    QTest.qWait(120)
    backend.timer.stop()
    for target in targets:
        assert getattr(target, method).call_count == 2
    parent.close()
    parent.deleteLater()
    app.processEvents()
