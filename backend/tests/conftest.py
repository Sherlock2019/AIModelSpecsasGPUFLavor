import os
import tempfile
from pathlib import Path

# Point the app at a throwaway database before any app module is imported.
_TMP = tempfile.mkdtemp(prefix="gpucalc-test-")
os.environ["DATABASE_URL"] = f"sqlite:///{Path(_TMP) / 'test.db'}"
os.environ.pop("ADMIN_TOKEN", None)

import pytest  # noqa: E402
import yaml  # noqa: E402

from app.engine import Catalog, default_settings  # noqa: E402
from app.engine.config import DATA_DIR  # noqa: E402
from app.schemas import AIFlavorDefinition, GPUSpec, ModelSpec, VGPUProfile  # noqa: E402


def _load(name, key):
    with open(DATA_DIR / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)[key]


@pytest.fixture(scope="session")
def settings():
    return default_settings()


@pytest.fixture(scope="session")
def models() -> dict[str, ModelSpec]:
    return {m["id"]: ModelSpec.model_validate(m) for m in _load("models.yaml", "models")}


@pytest.fixture(scope="session")
def catalog() -> Catalog:
    return Catalog(
        gpus=[GPUSpec.model_validate(g) for g in _load("gpus.yaml", "gpus")],
        profiles=[VGPUProfile.model_validate(p) for p in _load("vgpu_profiles.yaml", "vgpu_profiles")],
        ai_flavors=[AIFlavorDefinition.model_validate(f) for f in _load("ai_flavors.yaml", "ai_flavors")],
    )


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c
