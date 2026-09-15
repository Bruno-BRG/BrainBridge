"""
Training gateway backed by the existing ML training pipeline.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Optional
from uuid import uuid4

from brainbridge_v2.application.runtime_config import get_runtime
from brainbridge_v2.domain.entities.training_result import TrainingResult
from brainbridge_v2.infrastructure.config.settings import MODELS_DIR
from brainbridge_v2.infrastructure.ml import trainer as ml_trainer
from brainbridge_v2.infrastructure.ml.eeg_pipeline import pipeline_path, read_pipeline_manifest


class ModelTrainingGatewayAdapter:
    """
    Runs model training using the existing TensorFlow pipeline.
    """

    def __init__(
        self,
        trainer_module=ml_trainer,
        models_dir: Optional[Path] = None,
    ):
        self._trainer_module = trainer_module
        self._models_dir = models_dir or MODELS_DIR
        self._models_dir.mkdir(parents=True, exist_ok=True)

    def _patient_model_path(self, patient_id: int) -> Path:
        pointer = self._models_dir / f"patient_{patient_id}.json"
        if pointer.exists():
            name = json.loads(pointer.read_text(encoding="utf-8"))["model_path"]
            if (
                not isinstance(name, str)
                or Path(name).name != name
                or not name.startswith(f"patient_{patient_id}_candidate_")
                or not name.endswith(".keras")
            ):
                raise ValueError(f"Referencia de modelo invalida: {pointer}")
            selected = self._models_dir / name
            if not selected.is_file():
                raise FileNotFoundError(f"Modelo referenciado nao encontrado: {selected}")
            return selected
        return self._models_dir / f"patient_{patient_id}.keras"

    def _latest_base_model_path(self, emit: Callable[[str], None] = lambda _: None) -> Optional[Path]:
        candidates = [
            path
            for path in self._models_dir.glob("*.keras")
            if path.is_file() and not path.name.startswith("patient_")
        ]
        for path in sorted(candidates, key=lambda path: (path.stat().st_mtime, path.name), reverse=True):
            try:
                read_pipeline_manifest(path)
            except ValueError:
                emit(f"Ignorando modelo legado/incompativel: {path.name}. Nao sera usado para fine-tuning.")
                continue
            return path
        return None

    def _select_training_base_model(
        self,
        patient_id: int,
        emit: Callable[[str], None],
    ) -> Optional[Path]:
        patient_model = self._patient_model_path(patient_id)
        if patient_model.exists():
            try:
                read_pipeline_manifest(patient_model)
            except ValueError:
                # A published pointer is an explicit selection, not legacy discovery.
                if (self._models_dir / f"patient_{patient_id}.json").exists():
                    raise
                emit(f"Ignorando modelo legado/incompativel: {patient_model.name}. Nao sera usado para fine-tuning.")
            else:
                emit(f"Continuando treino do modelo do paciente: {patient_model.name}")
                return patient_model

        base_model = self._latest_base_model_path(emit)
        if base_model is not None:
            emit(f"Iniciando a partir do modelo base: {base_model.name}")
            return base_model

        emit("Nenhum modelo base compativel encontrado. Treinando modelo novo do zero.")
        return None

    def patient_model_available(self, patient_id: int) -> bool:
        """True se ha modelo publicado e valido para o paciente (calibracao feita)."""
        try:
            pid = int(patient_id)
        except (TypeError, ValueError):
            return False
        if pid <= 0:
            return False
        pointer = self._models_dir / f"patient_{pid}.json"
        if not pointer.exists():
            return False
        try:
            selected = self._patient_model_path(pid)
        except (ValueError, FileNotFoundError):
            return False
        if not selected.is_file():
            return False
        try:
            read_pipeline_manifest(selected)
        except ValueError:
            return False
        return True

    def train(
        self,
        csv_file_path: str,
        patient_id: int,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> TrainingResult:
        csv_path = Path(csv_file_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV de treinamento nao encontrado: {csv_file_path}")

        emit = progress_callback or (lambda _: None)

        emit("Preparando dados...")
        try:
            import tensorflow  # noqa: F401

        except Exception as exc:
            raise RuntimeError(
                "TensorFlow indisponivel: nao foi possivel iniciar o treinamento."
            ) from exc

        emit("Iniciando treinamento real (Keras)...")
        base_model_path = self._select_training_base_model(patient_id, emit)
        if base_model_path is not None:
            read_pipeline_manifest(base_model_path)
        base_accuracy = self._evaluate_base_on_csv(base_model_path, csv_path, emit)
        calib_epochs = int(get_runtime("calib_epochs", 15))
        calib_lr = get_runtime("calib_lr", 1e-4)
        calib_freeze = bool(get_runtime("calib_freeze_backbone", False))
        candidate = self._models_dir / f"patient_{patient_id}_candidate_{uuid4().hex}.keras"
        # The catalog only scans root files; partial saves stay private here.
        with TemporaryDirectory(prefix=".training-", dir=self._models_dir) as work_dir:
            staged = Path(work_dir).resolve() / candidate.name
            result = self._trainer_module.train_from_csvs(
                [str(csv_path)],
                model_name=str(staged.with_suffix("")),
                base_model_path=str(base_model_path) if base_model_path else None,
                epochs=calib_epochs,
                fine_tune_lr=float(calib_lr) if calib_lr else None,
                freeze_backbone=calib_freeze,
            )
            training_result = TrainingResult(
                model_path=str(candidate),
                training_time_seconds=float(result.training_time),
                final_accuracy=result.final_accuracy,
                final_loss=result.final_loss,
                val_accuracy=result.val_accuracy,
                val_loss=result.val_loss,
            )
            training_result.validate()
            if Path(result.model_path).resolve() != staged or not staged.is_file() or not staged.stat().st_size:
                raise ValueError("Treinamento nao produziu o arquivo de modelo esperado.")
            read_pipeline_manifest(staged)

            pointer = Path(work_dir) / "selected.json"
            pointer.write_text(json.dumps({"model_path": candidate.name}), encoding="utf-8")
            published = False
            manifest_published = False
            try:
                # Exclusive publication keeps previous candidates immutable.
                pipeline_path(candidate).hardlink_to(pipeline_path(staged))
                manifest_published = True
                candidate.hardlink_to(staged)
                published = True
                pointer.replace(self._models_dir / f"patient_{patient_id}.json")
            except BaseException:
                if published:
                    candidate.unlink()
                if manifest_published:
                    pipeline_path(candidate).unlink()
                raise
        if base_accuracy is not None and training_result.val_accuracy is not None:
            delta = (training_result.val_accuracy - base_accuracy) * 100.0
            emit(f"Delta calibracao: base {base_accuracy:.1%} -> novo {training_result.val_accuracy:.1%} ({delta:+.1f} p.p.).")
        elif training_result.val_accuracy is not None:
            emit(f"Nova acuracia (val interna): {training_result.val_accuracy:.1%}.")
        emit("Modelo salvo com sucesso.")
        return training_result

    def _evaluate_base_on_csv(self, base_model_path, csv_path, emit) -> Optional[float]:
        """Acuracia do modelo base nas janelas do CSV novo (delta visivel).

        Falhas nunca bloqueiam o treino (retorna None).
        """
        if base_model_path is None:
            return None
        try:
            import numpy as np

            data, markers = ml_trainer._load_openbci_csv(Path(csv_path))
            X, y = ml_trainer._create_windows_ht(data, markers)
            if len(X) == 0:
                return None
            model = ml_trainer.load_keras_model(str(base_model_path))
            probs = model.predict(np.asarray(X, dtype=np.float32),
                                  batch_size=64, verbose=0)
            pred = np.asarray(probs).argmax(axis=-1).reshape(-1)
            acc = float((pred == np.asarray(y).reshape(-1)).mean())
            emit(f"Acuracia base neste dado: {acc:.1%} ({len(y)} janelas).")
            return acc
        except Exception as exc:
            emit(f"Baseline da base indisponivel ({type(exc).__name__}); seguindo.")
            return None
