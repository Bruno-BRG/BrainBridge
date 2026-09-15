"""
Model catalog gateway backed by filesystem discovery.
"""

from pathlib import Path
from typing import Iterable, List, Optional

from brainbridge_v2.domain.entities.model_metadata import ModelMetadata
from brainbridge_v2.infrastructure.config.settings import MODELS_DIR


class FileSystemModelCatalogGatewayAdapter:
    """
    Lists available TensorFlow models from well-known project directories.
    """

    def __init__(self, search_roots: Optional[Iterable[Path]] = None):
        if search_roots is None:
            self._search_roots = None
        else:
            self._search_roots = [Path(root) for root in search_roots]

    def list_models(self) -> List[ModelMetadata]:
        discovered = {}

        for directory in self._candidate_dirs():
            if not directory.exists():
                continue

            for pattern in ("*.keras", "*.h5"):
                for path in directory.glob(pattern):
                    try:
                        resolved = path.resolve()
                        stat = resolved.stat()
                    except OSError:
                        continue

                    discovered[str(resolved)] = ModelMetadata(
                        path=str(resolved),
                        name=resolved.name,
                        modified_at_epoch=stat.st_mtime,
                    )

        return sorted(
            discovered.values(),
            key=lambda model: model.modified_at_epoch or 0.0,
            reverse=True,
        )

    def _candidate_dirs(self) -> List[Path]:
        if self._search_roots is not None:
            raw_dirs = list(self._search_roots)
        else:
            package_root = Path(__file__).resolve().parents[2]
            repo_root = Path(__file__).resolve().parents[3]
            cwd = Path.cwd()
            raw_dirs = [
                MODELS_DIR,
                repo_root / "bci" / "models",
                repo_root / "models",
                repo_root / "files",
                repo_root / "HardThinking" / "files",
                repo_root / "data" / "models",
                package_root / "data" / "models",
                cwd / "bci" / "models",
                cwd / "models",
                cwd / "files",
            ]

        unique_dirs = []
        seen = set()
        for directory in raw_dirs:
            normalized = str(directory.resolve())
            if normalized in seen:
                continue
            seen.add(normalized)
            unique_dirs.append(directory)
        return unique_dirs
