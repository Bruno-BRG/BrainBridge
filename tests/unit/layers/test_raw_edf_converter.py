import csv

import numpy as np
import pytest

mne = pytest.importorskip('mne')
from tools.downloader import transformador


@pytest.mark.parametrize('source_rate', [125, 160, 250])
def test_fake_edf_resamples_before_extraction_and_annotations(tmp_path, monkeypatch, source_rate):
    names = [f'EEG{i}' for i in range(16)]
    values = np.repeat(np.arange(1, 17)[:, None] * 1e-6, source_rate * 2, axis=1)
    raw = mne.io.RawArray(values, mne.create_info(names, source_rate, 'eeg'), verbose=False)
    raw.set_annotations(mne.Annotations([0, 1, 1.5], [0, 0, 0], ['T0', 'T1', 'T2']))
    original_resample = raw.resample
    calls = []
    def resample(rate):
        calls.append(rate)
        return original_resample(rate)
    monkeypatch.setattr(raw, 'resample', resample)
    monkeypatch.setattr(transformador.mne.io, 'read_raw_edf', lambda *args, **kwargs: raw)
    path = transformador.processar_edf_para_openbci(
        str(tmp_path / 'fake.edf'), None, names[::-1])
    with open(path) as source:
        lines = source.readlines()
    rows = list(csv.reader(line for line in lines if not line.startswith('%')))
    assert calls == [125.0]
    assert len(rows) == 251
    assert len(rows[0]) == 34
    assert '%Sample Rate = 125 Hz\n' in lines
    assert '%Signal Stage = raw\n' in lines
    assert any('Runtime Montage = unknown' in line for line in lines)
    assert f"%Channel Names = {', '.join(names[::-1])}\n" in lines
    np.testing.assert_allclose(np.array([r[1:17] for r in rows[1:]], dtype=float),
                               np.tile(np.arange(16, 0, -1), (250, 1)))
    assert rows[1][-1] == 'T0'
    assert rows[126][-1] == 'T1'
    assert rows[189][-1] == 'T2'
