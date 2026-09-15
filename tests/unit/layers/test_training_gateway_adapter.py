import tempfile
import os
import builtins
from pathlib import Path

import pytest
from brainbridge_v2.infrastructure.ml.eeg_pipeline import write_pipeline_manifest, read_pipeline_manifest, pipeline_path

from brainbridge_v2.infrastructure.ml.training_gateway_adapter import (
    ModelTrainingGatewayAdapter,
)
from brainbridge_v2.infrastructure.ml.model_catalog_gateway_adapter import (
    FileSystemModelCatalogGatewayAdapter,
)


class FakeTrainResult:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.training_time = 4.0
        self.final_accuracy = 0.77
        self.final_loss = 0.3
        self.val_accuracy = 0.71
        self.val_loss = 0.4


class FakeTrainerModule:
    def __init__(self):
        self.calls = []

    def train_from_csvs(self, csv_files, model_name, base_model_path=None, **kwargs):
        self.calls.append((csv_files, model_name, base_model_path, dict(kwargs)))
        path = Path(f"{model_name}.keras")
        path.write_bytes(b"trained candidate")
        write_pipeline_manifest(path, training_source_stages=["raw"])
        return FakeTrainResult(str(path))


def test_training_gateway_adapter_runs_real_training_when_tensorflow_is_available(monkeypatch):
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = Path(temp_dir)
        csv_path = tmp_path / "train.csv"
        csv_path.write_text("dummy", encoding="utf-8")
        trainer_module = FakeTrainerModule()
        monkeypatch.setitem(__import__("sys").modules, "tensorflow", object())

        adapter = ModelTrainingGatewayAdapter(
            trainer_module=trainer_module,
            models_dir=tmp_path,
        )
        progress_messages = []
        result = adapter.train(
            str(csv_path),
            5,
            progress_callback=progress_messages.append,
        )

        csv_files, model_name, base_model_path, _extra_kwargs = trainer_module.calls[0]
        assert csv_files == [str(csv_path)]
        assert Path(model_name).name.startswith("patient_5_candidate_")
        assert base_model_path is None
        assert read_pipeline_manifest(result.model_path)["training_source_stages"] == ["raw"]
        assert Path(result.model_path).name == f"{Path(model_name).name}.keras"
        assert progress_messages[:3] == [
            "Preparando dados...",
            "Iniciando treinamento real (Keras)...",
            "Nenhum modelo base compativel encontrado. Treinando modelo novo do zero.",
        ]


def test_training_gateway_adapter_continues_existing_patient_model(monkeypatch):
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = Path(temp_dir)
        csv_path = tmp_path / "train.csv"
        csv_path.write_text("dummy", encoding="utf-8")
        patient_model = tmp_path / "patient_5.keras"
        patient_model.write_text("existing patient model", encoding="utf-8")
        write_pipeline_manifest(patient_model, training_source_stages=["raw"])
        trainer_module = FakeTrainerModule()
        monkeypatch.setitem(__import__("sys").modules, "tensorflow", object())

        adapter = ModelTrainingGatewayAdapter(
            trainer_module=trainer_module,
            models_dir=tmp_path,
        )
        progress_messages = []
        adapter.train(
            str(csv_path),
            5,
            progress_callback=progress_messages.append,
        )

        assert len(trainer_module.calls) == 1
        assert Path(trainer_module.calls[0][1]).name.startswith("patient_5_candidate_")
        assert trainer_module.calls[0][2] == str(patient_model)
        assert progress_messages[2] == "Continuando treino do modelo do paciente: patient_5.keras"


def test_training_gateway_adapter_uses_latest_non_patient_model_as_initial_base(monkeypatch):
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = Path(temp_dir)
        csv_path = tmp_path / "train.csv"
        csv_path.write_text("dummy", encoding="utf-8")
        older_model = tmp_path / "generalized_old.keras"
        newer_model = tmp_path / "generalized_new.keras"
        older_model.write_text("older model", encoding="utf-8")
        newer_model.write_text("newer model", encoding="utf-8")
        write_pipeline_manifest(newer_model, training_source_stages=["raw"])
        os.utime(older_model, (100, 100))
        os.utime(newer_model, (200, 200))
        trainer_module = FakeTrainerModule()
        monkeypatch.setitem(__import__("sys").modules, "tensorflow", object())

        adapter = ModelTrainingGatewayAdapter(
            trainer_module=trainer_module,
            models_dir=tmp_path,
        )
        adapter.train(str(csv_path), 7)

        assert len(trainer_module.calls) == 1
        assert Path(trainer_module.calls[0][1]).name.startswith("patient_7_candidate_")
        assert trainer_module.calls[0][2] == str(newer_model)


@pytest.mark.parametrize("existing_model", [False, True])
@pytest.mark.parametrize("import_error", [ModuleNotFoundError, OSError])
def test_training_requires_tensorflow_without_writing_models(monkeypatch, existing_model, import_error):
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        csv_path = root / "train.csv"
        csv_path.write_text("dummy", encoding="utf-8")
        patient_model = root / "patient_5.keras"
        if existing_model:
            patient_model.write_bytes(b"original checkpoint")
        before = {path.name: path.read_bytes() for path in root.iterdir()}
        trainer = FakeTrainerModule()
        adapter = ModelTrainingGatewayAdapter(trainer_module=trainer, models_dir=root)
        original_import = builtins.__import__
        error = import_error("TensorFlow import failed")

        def fake_import(name, *args, **kwargs):
            if name == "tensorflow":
                raise error
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        messages = []
        with pytest.raises(RuntimeError, match="TensorFlow indisponivel") as raised:
            adapter.train(str(csv_path), 5, messages.append)

        assert raised.value.__cause__ is error
        assert trainer.calls == []
        assert messages == ["Preparando dados..."]
        assert {path.name: path.read_bytes() for path in root.iterdir()} == before


@pytest.mark.parametrize("failure", [None, "save", "validation", "pointer", "missing", "manifest"])
def test_training_candidates_preserve_existing_models(monkeypatch, failure):
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        csv_path = root / "train.csv"
        csv_path.write_text("dummy", encoding="utf-8")
        patient_model = root / "patient_5.keras"
        base_model = root / "generalized.keras"
        patient_model.write_bytes(b"patient checkpoint")
        base_model.write_bytes(b"base checkpoint")
        write_pipeline_manifest(base_model, training_source_stages=["raw"])
        write_pipeline_manifest(patient_model, training_source_stages=["raw"])
        monkeypatch.setitem(__import__("sys").modules, "tensorflow", object())

        catalog = FileSystemModelCatalogGatewayAdapter(search_roots=[root])
        initial = ModelTrainingGatewayAdapter(FakeTrainerModule(), root).train(str(csv_path), 5)
        pointer = root / "patient_5.json"
        previous_pointer = pointer.read_bytes()
        previous_catalog = {model.path for model in catalog.list_models()}
        expected_base = initial.model_path

        class SavingTrainer:
            def train_from_csvs(self, csv_files, model_name, base_model_path=None, **kwargs):
                assert base_model_path == expected_base
                path = root / f"{model_name}.keras"
                assert not path.exists()
                path.write_bytes(b"partial" if failure == "save" else b"trained candidate")
                write_pipeline_manifest(path, training_source_stages=["raw"])
                assert {model.path for model in catalog.list_models()} == previous_catalog
                if failure == "save":
                    raise OSError("save failed")
                result = FakeTrainResult(str(path))
                if failure == "validation":
                    result.training_time = -1
                if failure == "missing":
                    path.unlink()
                if failure == "manifest":
                    pipeline_path(path).unlink()
                return result

        if failure == "pointer":
            original_replace = Path.replace

            def fail_pointer_replace(self, target):
                if Path(target) == pointer:
                    raise OSError("pointer update failed")
                return original_replace(self, target)

            monkeypatch.setattr(Path, "replace", fail_pointer_replace)

        adapter = ModelTrainingGatewayAdapter(trainer_module=SavingTrainer(), models_dir=root)
        messages = []
        if failure:
            with pytest.raises(OSError if failure in ("save", "pointer") else ValueError):
                adapter.train(str(csv_path), 5, messages.append)
            assert "Modelo salvo com sucesso." not in messages
            assert pointer.read_bytes() == previous_pointer
            assert {model.path for model in catalog.list_models()} == previous_catalog
            assert ModelTrainingGatewayAdapter(FakeTrainerModule(), root)._patient_model_path(5) == Path(initial.model_path)
        else:
            first = adapter.train(str(csv_path), 5, messages.append)
            expected_base = first.model_path
            previous_catalog = {model.path for model in catalog.list_models()}
            adapter = ModelTrainingGatewayAdapter(trainer_module=SavingTrainer(), models_dir=root)
            second = adapter.train(str(csv_path), 5, messages.append)
            assert first.model_path != second.model_path
            assert Path(first.model_path).read_bytes() == b"trained candidate"
            assert Path(second.model_path).read_bytes() == b"trained candidate"
            assert messages[-1] == "Modelo salvo com sucesso."

        assert patient_model.read_bytes() == b"patient checkpoint"
        assert base_model.read_bytes() == b"base checkpoint"
        assert adapter._latest_base_model_path() == base_model
        assert Path(initial.model_path).read_bytes() == b"trained candidate"
        assert not list(root.glob(".training-*"))
        assert {p.stem for p in root.glob("patient_*candidate_*.keras")} == {
            p.name.removesuffix(".pipeline.json") for p in root.glob("patient_*candidate_*.pipeline.json")
        }


def test_training_rejects_missing_selected_checkpoint(monkeypatch):
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        csv_path = root / "train.csv"
        csv_path.write_text("dummy", encoding="utf-8")
        monkeypatch.setitem(__import__("sys").modules, "tensorflow", object())
        result = ModelTrainingGatewayAdapter(FakeTrainerModule(), root).train(str(csv_path), 5)
        Path(result.model_path).unlink()
        pointer = root / "patient_5.json"
        before = pointer.read_bytes()
        trainer = FakeTrainerModule()
        with pytest.raises(FileNotFoundError, match="Modelo referenciado"):
            ModelTrainingGatewayAdapter(trainer, root).train(str(csv_path), 5)
        assert not trainer.calls
        assert pointer.read_bytes() == before


@pytest.mark.parametrize("name", ["patient_5.keras", "generalized.keras"])
def test_legacy_only_trains_from_scratch_and_preserves_checkpoint(monkeypatch, tmp_path, name):
    monkeypatch.setitem(__import__("sys").modules, "tensorflow", object())
    csv_path = tmp_path / "train.csv"
    csv_path.write_text("dummy")
    legacy = tmp_path / name
    legacy.write_bytes(b"legacy checkpoint")
    trainer = FakeTrainerModule()
    messages = []
    result = ModelTrainingGatewayAdapter(trainer, tmp_path).train(str(csv_path), 5, messages.append)
    assert trainer.calls[0][2] is None
    assert any("Ignorando modelo legado/incompativel" in m and name in m for m in messages)
    assert any("Treinando modelo novo do zero" in m for m in messages)
    assert legacy.read_bytes() == b"legacy checkpoint"
    assert not pipeline_path(legacy).exists()
    assert Path(result.model_path) != legacy
    read_pipeline_manifest(result.model_path)


def test_auto_base_selection_skips_newer_incompatible_model(monkeypatch, tmp_path):
    monkeypatch.setitem(__import__("sys").modules, "tensorflow", object())
    csv_path = tmp_path / "train.csv"
    csv_path.write_text("dummy")
    compatible = tmp_path / "compatible.keras"
    incompatible = tmp_path / "incompatible.keras"
    compatible.write_bytes(b"compatible")
    incompatible.write_bytes(b"incompatible")
    write_pipeline_manifest(compatible, training_source_stages=["raw"])
    pipeline_path(incompatible).write_text('{"pipeline_version":"old"}')
    os.utime(compatible, (100, 100))
    os.utime(incompatible, (200, 200))
    trainer = FakeTrainerModule()
    messages = []
    ModelTrainingGatewayAdapter(trainer, tmp_path).train(str(csv_path), 5, messages.append)
    assert trainer.calls[0][2] == str(compatible)
    assert any("Ignorando" in m and incompatible.name in m for m in messages)


@pytest.mark.parametrize("failure", ["pointer", "manifest"])
def test_corrupt_published_selection_is_not_silently_replaced(monkeypatch, tmp_path, failure):
    monkeypatch.setitem(__import__("sys").modules, "tensorflow", object())
    csv_path = tmp_path / "train.csv"
    csv_path.write_text("dummy")
    gateway = ModelTrainingGatewayAdapter(FakeTrainerModule(), tmp_path)
    result = gateway.train(str(csv_path), 5)
    if failure == "pointer":
        (tmp_path / "patient_5.json").write_text("invalid json")
    else:
        pipeline_path(result.model_path).write_text("invalid json")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    trainer = FakeTrainerModule()
    with pytest.raises(ValueError):
        ModelTrainingGatewayAdapter(trainer, tmp_path).train(str(csv_path), 5)
    assert not trainer.calls
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
