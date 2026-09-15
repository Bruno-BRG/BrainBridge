import os
import tempfile
from pathlib import Path

from brainbridge_v2.infrastructure.ml.model_catalog_gateway_adapter import (
    FileSystemModelCatalogGatewayAdapter,
)


def test_model_catalog_gateway_adapter_lists_supported_files_sorted_by_mtime():
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = Path(temp_dir)
        older_model = tmp_path / "older.h5"
        latest_model = tmp_path / "latest.keras"
        ignored_file = tmp_path / "notes.txt"

        older_model.write_text("older", encoding="utf-8")
        latest_model.write_text("latest", encoding="utf-8")
        ignored_file.write_text("ignore", encoding="utf-8")

        os.utime(older_model, (10, 10))
        os.utime(latest_model, (20, 20))

        adapter = FileSystemModelCatalogGatewayAdapter(search_roots=[tmp_path])
        models = adapter.list_models()

        assert [model.name for model in models] == ["latest.keras", "older.h5"]
        assert models[0].path.endswith("latest.keras")


def test_default_catalog_includes_canonical_models_without_duplicates(monkeypatch, tmp_path):
    from brainbridge_v2.infrastructure.ml import model_catalog_gateway_adapter as catalog

    canonical = tmp_path / "canonical"
    canonical.mkdir()
    model = canonical / "trained.keras"
    model.write_bytes(b"checkpoint")
    monkeypatch.setattr(catalog, "MODELS_DIR", canonical)
    monkeypatch.chdir(tmp_path)
    # The cwd/models discovery path aliases the canonical training directory.
    (tmp_path / "models").symlink_to(canonical, target_is_directory=True)
    adapter = catalog.FileSystemModelCatalogGatewayAdapter()
    assert sum(p.resolve() == canonical.resolve() for p in adapter._candidate_dirs()) == 1
    assert sum(m.path == str(model.resolve()) for m in adapter.list_models()) == 1
