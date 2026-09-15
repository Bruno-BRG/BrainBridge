import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PyQt5.QtCore import Qt

from brainbridge_v2.infrastructure.acquisition import streaming_thread as streaming


@pytest.mark.parametrize('encoding', ['timeseries', 'channels', 'json'])
def test_extract_preserves_every_raw_sample_and_channel(encoding):
    values = np.arange(16 * 250, dtype=float).reshape(16, 250) + 10000.25
    packet = {'type': 'timeSeriesRaw', 'data': values.tolist()}
    if encoding == 'channels':
        packet = {f'Ch{i + 1}': values[i].tolist() for i in range(16)}
    elif encoding == 'json':
        packet = json.dumps(packet)
    thread = streaming.StreamingThread()
    result = thread.extract_eeg_from_udp(packet)
    np.testing.assert_array_equal(result, values.T)
    assert thread.sample_rate == 125
    assert not hasattr(thread, 'butter_filter')


@pytest.mark.parametrize('packet', [list(range(15)), {'Ch1': 1},
    {'type': 'timeSeriesRaw', 'data': [[1, 2]] * 15 + [[3]]}])
def test_incomplete_packets_are_not_padded(packet):
    assert streaming.StreamingThread().extract_eeg_from_udp(packet) is None


def test_single_raw_sample():
    values = [float(i * 100) for i in range(16)]
    thread = streaming.StreamingThread()
    np.testing.assert_array_equal(thread.extract_eeg_from_udp(values), values)
    np.testing.assert_array_equal(thread.extract_eeg_from_udp(
        {f'Ch{i + 1}': value for i, value in enumerate(values)}), values)


@pytest.mark.parametrize('raises', [False, True])
def test_start_failure_never_emits_fake_data(monkeypatch, raises):
    receiver = SimpleNamespace(is_running=False, socket=Mock(), stop=Mock(),
                               set_callback=Mock(), start=Mock())
    if raises:
        receiver.start.side_effect = OSError('port unavailable')
    monkeypatch.setattr(streaming, 'UDPReceiver_BCI', lambda *args: receiver)
    thread = streaming.StreamingThread()
    thread.host, thread.port, thread.is_running = 'localhost', 12345, True
    samples, statuses = [], []
    thread.data_received.connect(samples.append, Qt.DirectConnection)
    thread.connection_status.connect(statuses.append, Qt.DirectConnection)
    thread.run()
    assert samples == []
    assert statuses == [False]
    assert thread.last_error
    assert not thread.is_running
    assert not thread.is_mock_mode
    assert receiver.socket is None


def test_run_emits_complete_raw_batch(monkeypatch):
    thread = streaming.StreamingThread()
    thread.host, thread.port, thread.is_running = 'localhost', 12345, True
    values = np.arange(80).reshape(16, 5)
    receiver = SimpleNamespace(is_running=True, socket=None, stop=Mock())
    receiver.set_callback = lambda callback: setattr(receiver, 'callback', callback)
    def start():
        receiver.callback({'type': 'timeSeriesRaw', 'data': values.tolist()})
        thread.is_running = False
    receiver.start = start
    monkeypatch.setattr(streaming, 'UDPReceiver_BCI', lambda *args: receiver)
    samples = []
    thread.data_received.connect(samples.append, Qt.DirectConnection)
    thread.run()
    np.testing.assert_array_equal(samples, values.T)
