import numpy as np

from brainbridge_v2.application.free_run_coordinator import FreeRunInferenceCoordinator
from brainbridge_v2.infrastructure.ml.eeg_pipeline import (
    adapt_channel_count,
    preprocess_window_adaptive,
    resample_window,
)


def test_resample_128_to_125():
    rng = np.random.default_rng(0)
    window = rng.normal(size=(256, 16))
    out = resample_window(window, input_fs=128.0, target_fs=125.0)
    assert out.shape == (250, 16)


def test_adapt_channels_truncate_and_pad():
    rng = np.random.default_rng(1)
    wide = rng.normal(size=(250, 22))
    assert adapt_channel_count(wide).shape == (250, 16)
    narrow = rng.normal(size=(250, 3))
    out = adapt_channel_count(narrow)
    assert out.shape == (250, 16)
    assert (out[:, 3:] == 0).all()


def test_adapt_channels_by_name():
    rng = np.random.default_rng(2)
    names = ["C3", "C4", "Cz", "Pz"] + [f"X{i}" for i in range(12)]
    window = rng.normal(size=(100, 16))
    out = adapt_channel_count(window, channel_names=names,
                              target_names=["C3", "C4"])
    assert out.shape == (100, 2)
    np.testing.assert_array_equal(out[:, 0], window[:, 0])
    np.testing.assert_array_equal(out[:, 1], window[:, 1])


def test_preprocess_adaptive_accepts_128_250_and_any_channels():
    rng = np.random.default_rng(3)
    for fs, t, c in ((128.0, 256, 8), (250.0, 500, 16), (125.0, 250, 22)):
        window = rng.normal(size=(t, c))
        out = preprocess_window_adaptive(window, input_fs=fs)
        assert out.shape == (250, 16)
        assert out.dtype == np.float32
        assert np.isfinite(out).all()


def test_free_run_emits_with_stride():
    coord = FreeRunInferenceCoordinator(window_size=250, channels=16, stride=125)
    coord.start()
    ready = 0
    for _ in range(500):
        res = coord.add_sample([0.0] * 16)
        if res.status == FreeRunInferenceCoordinator.STATUS_READY:
            ready += 1
            assert len(res.window) == 250
    assert ready == 3


def test_free_run_inactive_until_start():
    coord = FreeRunInferenceCoordinator()
    res = coord.add_sample([0.0] * 16)
    assert res.status == FreeRunInferenceCoordinator.STATUS_INACTIVE


def test_balance_caps_and_augments_minority():
    import collections
    from brainbridge_v2.infrastructure.ml.trainer import _balance_and_augment_train

    rng = np.random.default_rng(0)
    X = rng.normal(size=(2500, 250, 16)).astype(np.float32)
    y = np.array([0] * 1000 + [1] * 1000 + [0] * 250 + [1] * 250)
    g = np.array(["A"] * 2000 + ["B"] * 500)
    Xb, yb, gb = _balance_and_augment_train(X, y, g, cap_per_group=400, augment=True)
    counts = collections.Counter(zip(gb.tolist(), yb.tolist()))
    assert Xb.shape == (800, 250, 16)
    assert counts[("A", 0)] == counts[("A", 1)] == 200
    assert counts[("B", 0)] == counts[("B", 1)] == 200
    # Sem augment: ambos os grupos limitados ao cap estratificado.
    Xb2, yb2, gb2 = _balance_and_augment_train(X, y, g, cap_per_group=400, augment=False)
    assert len(Xb2) == 400 + 400


def test_ea_whitens_group_mean_covariance_to_identity():
    from brainbridge_v2.infrastructure.ml.eeg_pipeline import (
        apply_ea, bandpass_window, ea_reference_matrix, ea_whitening_matrix,
        preprocess_ea_window)

    rng = np.random.default_rng(0)
    # Dois "sujeitos" com covariancias bem diferentes (media zero, pos-filtro).
    A = rng.normal(size=(10, 250, 16)) * 5
    A = A - A.mean(axis=1, keepdims=True)
    B = rng.normal(size=(10, 250, 16)) * 0.5
    B = B - B.mean(axis=1, keepdims=True)
    for X in (A, B):
        W = ea_whitening_matrix(ea_reference_matrix(X))
        Xa = apply_ea(X, W)
        C = np.einsum("nti,ntj->ij", Xa, Xa) / (len(Xa) * 250)
        np.testing.assert_allclose(np.diag(C), 1.0, atol=0.05)
        assert abs(C - np.diag(np.diag(C))).max() < 0.05
    # Pipeline fechado EA: (250,16) @125Hz, com e sem reamostragem.
    out = preprocess_ea_window(A[0], ea_whitening_matrix(ea_reference_matrix(A)))
    assert out.shape == (250, 16) and out.dtype == np.float32
    out128 = preprocess_ea_window(
        rng.normal(size=(256, 16)),
        np.eye(16), input_fs=128.0)
    assert out128.shape == (250, 16)


def test_ea_session_aligner_calibrates_and_applies():
    from brainbridge_v2.infrastructure.ml.eeg_pipeline import EASessionAligner

    rng = np.random.default_rng(1)
    aligner = EASessionAligner(channels=4, required_samples=100)
    assert not aligner.is_ready
    for _ in range(99):
        assert aligner.observe(rng.normal(size=4)) is False
    assert aligner.observe(rng.normal(size=4)) is True
    assert aligner.is_ready
    assert aligner.progress() == (100, 100)
    out = aligner.apply(np.zeros((250, 4)))
    assert out.shape == (250, 4)


def test_ea_manifest_flag_roundtrip(tmp_path, monkeypatch):
    import sys
    from pathlib import Path as _P
    from brainbridge_v2.infrastructure.ml.eeg_pipeline import (
        manifest_uses_ea, pipeline_path, write_pipeline_manifest)

    monkeypatch.setitem(sys.modules, "tensorflow", object())
    model = tmp_path / "m.keras"
    model.write_bytes(b"fake")
    write_pipeline_manifest(model, training_source_stages=["raw"],
                            euclidean_alignment=True)
    assert manifest_uses_ea(model) is True
    assert "euclidean_alignment" in pipeline_path(model).read_text()
    write_pipeline_manifest(model, training_source_stages=["raw"])
    assert manifest_uses_ea(model) is False
