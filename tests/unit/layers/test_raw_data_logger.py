import csv
from datetime import datetime

import pytest

from brainbridge_v2.infrastructure.acquisition import data_logger


def test_raw_logger_preserves_schema_values_and_host_timestamps(tmp_path, monkeypatch):
    times = iter([1700000000.125, 1700000000.391])
    monkeypatch.setattr(data_logger.time, 'time', lambda: next(times))
    logger = data_logger.OpenBCICSVLogger('1', 'test', base_path=str(tmp_path))
    values = [10000.125 + i for i in range(16)]
    logger.log_sample(values, 'T1')
    logger.log_sample(values)
    logger.close()
    with open(logger.get_full_path()) as source:
        lines = source.readlines()
    assert '%Signal Stage = raw\n' in lines
    assert '%Sample Rate = 125 Hz\n' in lines
    assert any('not device acquisition time' in line for line in lines)
    assert any('Montage = unknown' in line for line in lines)
    rows = list(csv.reader(line for line in lines if not line.startswith('%')))
    header, first, second = rows
    assert len(header) == len(first) == len(second) == 34
    assert header[1:17] == [f'EXG Channel {i}' for i in range(16)]
    assert header[30:] == ['Timestamp', 'Other.7', 'Timestamp (Formatted)', 'Annotations']
    assert list(map(float, first[1:17])) == values
    assert float(first[30]) == 1700000000.125
    assert float(second[30]) == 1700000000.391
    assert first[32] == datetime.fromtimestamp(1700000000.125).isoformat(timespec='milliseconds')
    assert first[33] == 'T1'
    assert second[33] == ''


def test_logger_rejects_partial_cap_and_preserves_marker_duration(tmp_path):
    logger = data_logger.OpenBCICSVLogger('1', 'test', base_path=str(tmp_path))
    with pytest.raises(ValueError, match='16 canais'):
        logger.log_sample([1] * 8)
    logger.log_sample([1] * 16, 'T2')
    for _ in range(250):
        logger.log_sample([2] * 16)
    logger.close()
    with open(logger.get_full_path()) as source:
        rows = list(csv.DictReader(line for line in source if not line.startswith('%')))
    assert len(rows) == 251
    assert rows[-2]['Annotations'] == ''
    assert rows[-1]['Annotations'] == 'T0'
