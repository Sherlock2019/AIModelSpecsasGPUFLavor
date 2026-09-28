"""Catalog persistence: seeding from YAML, CRUD, settings and engine snapshots."""

from __future__ import annotations

from typing import Optional

import yaml
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .db import (
    AIFlavorRow,
    BenchmarkRow,
    DeploymentSpecRow,
    GPURow,
    InventoryRow,
    ModelRow,
    SavedCalculationRow,
    SettingRow,
    VGPUProfileRow,
)
from .engine.calculator import Catalog
from .engine.config import DATA_DIR, EngineSettings, load_default_settings_dict
from .inventory import StaticInventoryAdapter
from .schemas import AIFlavorDefinition, BenchmarkRecord, GPUSpec, InventoryRecord, ModelSpec, VGPUProfile

ENGINE_SETTINGS_KEY = "engine"


def _load_yaml(name: str, key: str) -> list[dict]:
    with open(DATA_DIR / name, encoding="utf-8") as fh:
        return (yaml.safe_load(fh) or {}).get(key, [])


# ---------------------------------------------------------------------------- seeding


def seed(session: Session) -> None:
    """Add bundled catalog entries that are missing (by id). Never overwrites existing rows, so admin
    edits survive upgrades while new models/GPUs shipped with a release still appear.

    Existing ids are read once per table and new rows are added in one transaction (a per-row lookup
    auto-flushes and makes a fresh database slow to seed on slow disks)."""
    with session.no_autoflush:
        have = set(session.scalars(select(ModelRow.id)))
        for raw in _load_yaml("models.yaml", "models"):
            if raw["id"] not in have:
                spec = ModelSpec.model_validate(raw)
                session.add(ModelRow(id=spec.id, name=spec.name, family=spec.family, enabled=spec.enabled,
                                     metadata_status=spec.metadata_status, spec=spec.model_dump(mode="json")))
        have = set(session.scalars(select(GPURow.id)))
        for raw in _load_yaml("gpus.yaml", "gpus"):
            if raw["id"] not in have:
                spec = GPUSpec.model_validate(raw)
                session.add(GPURow(id=spec.id, enabled=spec.enabled, spec=spec.model_dump(mode="json")))
        have = set(session.scalars(select(VGPUProfileRow.profile_name)))
        for raw in _load_yaml("vgpu_profiles.yaml", "vgpu_profiles"):
            if raw["profile_name"] not in have:
                spec = VGPUProfile.model_validate(raw)
                session.add(VGPUProfileRow(profile_name=spec.profile_name, physical_gpu=spec.physical_gpu,
                                           enabled=spec.enabled, spec=spec.model_dump(mode="json")))
        have = set(session.scalars(select(AIFlavorRow.id)))
        for raw in _load_yaml("ai_flavors.yaml", "ai_flavors"):
            if raw["id"] not in have:
                spec = AIFlavorDefinition.model_validate(raw)
                session.add(AIFlavorRow(id=spec.id, enabled=spec.enabled, spec=spec.model_dump(mode="json")))
    if session.scalar(select(InventoryRow.id).limit(1)) is None:
        for raw in _load_yaml("inventory.yaml", "inventory"):
            rec = InventoryRecord.model_validate(raw)
            session.add(InventoryRow(gpu_id=rec.gpu_id, site=rec.site, installed=rec.installed, available=rec.available))
    if session.get(SettingRow, ENGINE_SETTINGS_KEY) is None:
        session.add(SettingRow(key=ENGINE_SETTINGS_KEY, value=load_default_settings_dict()))
    session.commit()


# ---------------------------------------------------------------------------- models


def _model_from_row(row: ModelRow) -> ModelSpec:
    spec = dict(row.spec)
    spec["enabled"] = row.enabled
    return ModelSpec.model_validate(spec)


def list_models(session: Session, include_disabled: bool = False) -> list[ModelSpec]:
    stmt = select(ModelRow).order_by(ModelRow.family, ModelRow.name)
    if not include_disabled:
        stmt = stmt.where(ModelRow.enabled.is_(True))
    return [_model_from_row(r) for r in session.scalars(stmt)]


def get_model(session: Session, model_id: str) -> Optional[ModelSpec]:
    row = session.get(ModelRow, model_id)
    return _model_from_row(row) if row else None


def upsert_model(session: Session, spec: ModelSpec, commit: bool = True) -> ModelSpec:
    data = spec.model_dump(mode="json")
    row = session.get(ModelRow, spec.id)
    if row is None:
        row = ModelRow(id=spec.id)
        session.add(row)
    row.name, row.family, row.enabled = spec.name, spec.family, spec.enabled
    row.metadata_status = spec.metadata_status
    row.spec = data
    if commit:
        session.commit()
    return spec


def export_models_yaml(session: Session) -> str:
    models = [m.model_dump(mode="json", exclude_none=True) for m in list_models(session, include_disabled=True)]
    return yaml.safe_dump({"models": models}, sort_keys=False, allow_unicode=True)


def import_models_yaml(session: Session, text: str, overwrite: bool) -> dict:
    """Parse YAML (safe_load only - data, never code) and upsert valid models."""
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML: {exc}") from exc
    items = doc.get("models") if isinstance(doc, dict) else doc
    if not isinstance(items, list):
        raise ValueError("Expected a list of models or a mapping with a 'models' key")
    created = updated = skipped = 0
    errors: list[str] = []
    for i, raw in enumerate(items):
        try:
            spec = ModelSpec.model_validate(raw)
        except Exception as exc:  # pydantic ValidationError or bad type
            errors.append(f"item {i}: {exc}")
            continue
        exists = session.get(ModelRow, spec.id) is not None
        if exists and not overwrite:
            skipped += 1
            continue
        upsert_model(session, spec, commit=False)
        updated += exists
        created += not exists
    session.commit()
    return {"created": created, "updated": updated, "skipped": skipped, "errors": errors}


# ---------------------------------------------------------------------------- GPUs


def _gpu_from_row(row: GPURow) -> GPUSpec:
    spec = dict(row.spec)
    spec["enabled"] = row.enabled
    return GPUSpec.model_validate(spec)


def list_gpus(session: Session, include_disabled: bool = True) -> list[GPUSpec]:
    stmt = select(GPURow)
    if not include_disabled:
        stmt = stmt.where(GPURow.enabled.is_(True))
    gpus = [_gpu_from_row(r) for r in session.scalars(stmt)]
    return sorted(gpus, key=lambda g: (g.vram_gb, g.memory_bandwidth_gbps or 0))


def get_gpu(session: Session, gpu_id: str) -> Optional[GPUSpec]:
    row = session.get(GPURow, gpu_id)
    return _gpu_from_row(row) if row else None


def upsert_gpu(session: Session, spec: GPUSpec, commit: bool = True) -> GPUSpec:
    row = session.get(GPURow, spec.id)
    if row is None:
        row = GPURow(id=spec.id)
        session.add(row)
    row.enabled = spec.enabled
    row.spec = spec.model_dump(mode="json")
    if commit:
        session.commit()
    return spec


def list_profiles(session: Session) -> list[VGPUProfile]:
    rows = session.scalars(select(VGPUProfileRow))
    profiles = []
    for row in rows:
        spec = dict(row.spec)
        spec["enabled"] = row.enabled
        profiles.append(VGPUProfile.model_validate(spec))
    return sorted(profiles, key=lambda p: (p.physical_gpu, p.sharing_mode, p.vram_gb))


def upsert_profile(session: Session, spec: VGPUProfile, commit: bool = True) -> VGPUProfile:
    row = session.get(VGPUProfileRow, spec.profile_name)
    if row is None:
        row = VGPUProfileRow(profile_name=spec.profile_name)
        session.add(row)
    row.physical_gpu = spec.physical_gpu
    row.enabled = spec.enabled
    row.spec = spec.model_dump(mode="json")
    if commit:
        session.commit()
    return spec


# ---------------------------------------------------------------------------- inventory


def list_inventory(session: Session) -> list[InventoryRecord]:
    rows = session.scalars(select(InventoryRow).order_by(InventoryRow.site, InventoryRow.gpu_id))
    return [
        InventoryRecord(id=r.id, gpu_id=r.gpu_id, site=r.site, installed=r.installed, available=r.available) for r in rows
    ]


def replace_inventory(session: Session, records: list[InventoryRecord]) -> list[InventoryRecord]:
    session.execute(delete(InventoryRow))
    for rec in records:
        session.add(InventoryRow(gpu_id=rec.gpu_id, site=rec.site, installed=rec.installed, available=rec.available))
    session.commit()
    return list_inventory(session)


# ---------------------------------------------------------------------------- benchmarks


def list_benchmarks(session: Session, model_id: Optional[str] = None, gpu_id: Optional[str] = None) -> list[BenchmarkRecord]:
    stmt = select(BenchmarkRow).order_by(BenchmarkRow.id.desc())
    if model_id:
        stmt = stmt.where(BenchmarkRow.model_id == model_id)
    if gpu_id:
        stmt = stmt.where(BenchmarkRow.gpu_id == gpu_id)
    out = []
    for row in session.scalars(stmt):
        spec = dict(row.spec)
        spec["id"] = row.id
        out.append(BenchmarkRecord.model_validate(spec))
    return out


def add_benchmark(session: Session, rec: BenchmarkRecord) -> BenchmarkRecord:
    row = BenchmarkRow(model_id=rec.model_id, gpu_id=rec.gpu_id, precision=rec.precision, spec=rec.model_dump(mode="json", exclude={"id"}))
    session.add(row)
    session.commit()
    return rec.model_copy(update={"id": row.id})


def delete_benchmark(session: Session, bench_id: int) -> bool:
    row = session.get(BenchmarkRow, bench_id)
    if row is None:
        return False
    session.delete(row)
    session.commit()
    return True


# ---------------------------------------------------------------------------- settings


def get_settings_dict(session: Session) -> dict:
    row = session.get(SettingRow, ENGINE_SETTINGS_KEY)
    return row.value if row else load_default_settings_dict()


def get_settings(session: Session) -> EngineSettings:
    return EngineSettings.model_validate(get_settings_dict(session))


def put_settings(session: Session, value: dict) -> EngineSettings:
    settings = EngineSettings.model_validate(value)  # validate before persisting
    row = session.get(SettingRow, ENGINE_SETTINGS_KEY)
    data = settings.model_dump(mode="json")
    if row is None:
        session.add(SettingRow(key=ENGINE_SETTINGS_KEY, value=data))
    else:
        row.value = data
    session.commit()
    return settings


def reset_settings(session: Session) -> EngineSettings:
    return put_settings(session, load_default_settings_dict())


# ---------------------------------------------------------------------------- saved calculations


def save_calculation(session: Session, name: str, request: dict, result: dict) -> SavedCalculationRow:
    row = SavedCalculationRow(name=name, request=request, result=result)
    session.add(row)
    session.commit()
    return row


def list_calculations(session: Session) -> list[SavedCalculationRow]:
    return list(session.scalars(select(SavedCalculationRow).order_by(SavedCalculationRow.created_at.desc())))


def get_calculation(session: Session, calc_id: int) -> Optional[SavedCalculationRow]:
    return session.get(SavedCalculationRow, calc_id)


def delete_calculation(session: Session, calc_id: int) -> bool:
    row = session.get(SavedCalculationRow, calc_id)
    if row is None:
        return False
    session.delete(row)
    session.commit()
    return True


# ---------------------------------------------------------------------------- AI flavors


def list_flavors(session: Session, include_disabled: bool = True) -> list[AIFlavorDefinition]:
    out = []
    for row in session.scalars(select(AIFlavorRow)):
        spec = dict(row.spec)
        spec["enabled"] = row.enabled
        out.append(AIFlavorDefinition.model_validate(spec))
    out.sort(key=lambda f: f.parameter_ceiling_b)
    return [f for f in out if include_disabled or f.enabled]


def get_flavor(session: Session, flavor_id: str) -> Optional[AIFlavorDefinition]:
    row = session.get(AIFlavorRow, flavor_id)
    if row is None:
        return None
    spec = dict(row.spec)
    spec["enabled"] = row.enabled
    return AIFlavorDefinition.model_validate(spec)


def upsert_flavor(session: Session, spec: AIFlavorDefinition, commit: bool = True) -> AIFlavorDefinition:
    row = session.get(AIFlavorRow, spec.id)
    if row is None:
        row = AIFlavorRow(id=spec.id)
        session.add(row)
    row.enabled = spec.enabled
    row.spec = spec.model_dump(mode="json")
    if commit:
        session.commit()
    return spec


def delete_flavor(session: Session, flavor_id: str) -> bool:
    row = session.get(AIFlavorRow, flavor_id)
    if row is None:
        return False
    session.delete(row)
    session.commit()
    return True


# ---------------------------------------------------------------------------- deployment specs


def save_deployment_spec(session: Session, name: Optional[str], spec: dict, openstack: dict) -> DeploymentSpecRow:
    row = DeploymentSpecRow(
        name=name,
        workload_type=str(spec.get("workload_type", "llm")),
        ai_flavor=str(spec.get("ai_flavor", "")),
        spec=spec,
        openstack=openstack,
    )
    session.add(row)
    session.commit()
    return row


def list_deployment_specs(session: Session) -> list[DeploymentSpecRow]:
    return list(session.scalars(select(DeploymentSpecRow).order_by(DeploymentSpecRow.created_at.desc())))


def delete_deployment_spec(session: Session, spec_id: int) -> bool:
    row = session.get(DeploymentSpecRow, spec_id)
    if row is None:
        return False
    session.delete(row)
    session.commit()
    return True


# ---------------------------------------------------------------------------- engine snapshot


def catalog_snapshot(session: Session, site: Optional[str] = None) -> Catalog:
    adapter = StaticInventoryAdapter(session, site=site)
    capacity = adapter.get_gpu_capacity()
    return Catalog(
        gpus=adapter.get_available_gpus(),
        profiles=adapter.get_vgpu_profiles(),
        capacity=capacity,
        has_inventory=bool(capacity),
        benchmarks=list_benchmarks(session),
        ai_flavors=list_flavors(session, include_disabled=False),
    )
