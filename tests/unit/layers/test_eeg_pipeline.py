import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

from brainbridge_v2.infrastructure.ml.eeg_pipeline import (
    preprocess_window, pipeline_path, read_pipeline_manifest, write_pipeline_manifest,
)
from brainbridge_v2.infrastructure.ml import trainer


@pytest.fixture
def tmp_path():
    with tempfile.TemporaryDirectory() as directory:
        yield Path(directory)


def test_numerical_reference_and_band_response():
    t = np.arange(250) / 125
    wave = sum(np.sin(2 * np.pi * f * t) for f in (3, 15, 45))
    window = np.tile(wave[:, None], (1, 16))
    filtered = sosfiltfilt(butter(6, (8, 30), fs=125, btype="bandpass", output="sos"), window, axis=0)
    expected = ((filtered - filtered.mean(0)) / (filtered.std(0) + 1e-6)).astype(np.float32)
    actual = preprocess_window(window)
    np.testing.assert_array_equal(actual, expected)
    spectrum = abs(np.fft.rfft(actual[:, 0]))
    assert spectrum[30] > 10 * max(spectrum[6], spectrum[90])
    assert actual.dtype == np.float32


@pytest.mark.parametrize("window", [np.zeros((249, 16)), np.zeros((250, 15)), np.zeros((1, 250, 16)), np.full((250, 16), np.nan), np.full((250, 16), np.inf)])
def test_strict_shape_finite(window):
    with pytest.raises(ValueError):
        preprocess_window(window)


def test_constant_and_configuration():
    np.testing.assert_allclose(preprocess_window(np.zeros((250, 16))), 0)
    for kwargs in ({"sample_rate": 250}, {"band": (1, 30)}):
        with pytest.raises(ValueError):
            preprocess_window(np.zeros((250, 16)), **kwargs)


def test_markers_parity_overlap_and_trial_split():
    data = np.random.default_rng(7).normal(size=(2501, 16))
    markers = [""] * len(data)
    for i, marker in ((0, "T1"), (500, "T2"), (1000, "T1"), (1500, "T2"), (2000, "other"), (2250, "T1")):
        markers[i] = marker
    X, y, groups = trainer._create_windows_ht(data, markers, return_groups=True)
    assert X.shape == (12, 250, 16)  # unknown marker closes T2; final trial is incomplete
    np.testing.assert_array_equal(X[0], preprocess_window(data[:250]))
    np.testing.assert_array_equal(X[1], preprocess_window(data[125:375]))
    train, val = trainer._split_trials(y, groups)
    assert set(groups[train]).isdisjoint(groups[val])
    assert set(y[train]) == set(y[val]) == {0, 1}
    duplicate_groups = trainer._create_windows_ht(data.copy(), markers, return_groups=True)[2]
    np.testing.assert_array_equal(groups, duplicate_groups)
    train2, val2 = trainer._split_trials(np.tile(y, 2), np.tile(groups, 2))
    assert set(np.tile(groups, 2)[train2]).isdisjoint(np.tile(groups, 2)[val2])
    with pytest.raises(ValueError, match="two distinct trials"):
        trainer._split_trials(np.array([0, 1]), np.array(["a", "b"]))


def write_csv(path, headers="%Sample Rate = 125 Hz\n%Signal Stage = raw\n"):
    rng = np.random.default_rng(9)
    rows = []
    for i, row in enumerate(rng.normal(size=(1501, 16))):
        marker = {0: "T1", 375: "T2", 750: "T1", 1125: "T2", 1500: "T0"}.get(i, "")
        rows.append(",".join([str(i), *map(str, row), marker]))
    path.write_text(headers + "\n".join(rows))


def test_csv_provenance(tmp_path):
    path = tmp_path / "data.csv"
    write_csv(path)
    stages = []
    trainer._load_openbci_csv(path, source_stages=stages)
    assert stages == ["raw"]
    # 250/128 Hz sao aceitos via reamostragem adaptativa para 125 Hz.
    for header in ("%Sample Rate = 250 Hz\n%Signal Stage = raw\n",
                   "%Sample Rate = 128 Hz\n%Signal Stage = raw\n"):
        write_csv(path, header)
        data, markers = trainer._load_openbci_csv(path)
        assert data.shape[1] == 16 and len(markers) == len(data)
    # Stage filtrado continua rejeitado; fs absurda rejeitada.
    for header in ("%Signal Stage = filtered\n",
                   "%Sample Rate = 5000 Hz\n%Signal Stage = raw\n"):
        write_csv(path, header)
        with pytest.raises(ValueError):
            trainer._load_openbci_csv(path)
    write_csv(path, "")
    with pytest.warns(RuntimeWarning, match="unverified"):
        trainer._load_openbci_csv(path)


def test_real_training_flow_fake_tf_manifest_and_legacy_rejection(tmp_path, monkeypatch):
    from tests.unit.layers.test_generalized_training import FakeGeneralizedModel
    monkeypatch.setitem(sys.modules, "tensorflow", object())
    monkeypatch.setattr(trainer, "MODELS_DIR", tmp_path)
    path = tmp_path / "data.csv"
    write_csv(path)
    duplicate = tmp_path / "copy.csv"
    duplicate.write_bytes(path.read_bytes())
    model = FakeGeneralizedModel()
    result = trainer.train_from_csvs([str(path), str(duplicate)], model_name="new", model_builder=lambda *args: model)
    assert read_pipeline_manifest(result.model_path)["training_source_stages"] == ["raw"]
    args, kwargs = model.fit_calls[0]
    assert "validation_split" not in kwargs
    assert set(args[1]) == set(kwargs["validation_data"][1]) == {0, 1}
    assert not {w.tobytes() for w in args[0]} & {w.tobytes() for w in kwargs["validation_data"][0]}
    pipeline_path(result.model_path).unlink()
    with pytest.raises(ValueError, match="retrain"):
        trainer.train_from_csvs([str(path)], base_model_path=result.model_path, model_loader=lambda *args: pytest.fail("must not load legacy"))


def test_manifest_incompatible(tmp_path):
    path = tmp_path / "model.keras"
    write_pipeline_manifest(path, training_source_stages=["unknown"])
    assert read_pipeline_manifest(path)["training_source_stages"] == ["unknown"]
    pipeline_path(path).write_text('{"pipeline_version": "old"}')
    with pytest.raises(ValueError, match="retrain"):
        read_pipeline_manifest(path)


def test_generalized_rejects_single_class_partitions(monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorflow", object())
    monkeypatch.setattr(trainer, "_collect_windowed_dataset", lambda *a, **k: (
        np.zeros((4, 250, 16)), np.array([0, 0, 1, 1]),
        np.array(["P1", "P1", "P2", "P2"]), [],
    ))
    with pytest.raises(ValueError, match="Both train and validation"):
        trainer.train_generalized_from_csvs(["unused"])


def test_generalized_duplicate_subject_rejected(tmp_path):
    path = tmp_path / "one.csv"
    copy = tmp_path / "two.csv"
    write_csv(path)
    copy.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="Duplicate CSV"):
        trainer.load_generalized_windowed_dataset(
            [str(path), str(copy)], left_right_only=False,
            group_resolver=lambda p: Path(p).stem,
        )
