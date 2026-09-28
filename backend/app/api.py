"""REST API (mounted under /api/v1)."""

from __future__ import annotations

import os
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response
from pydantic import ValidationError
from sqlalchemy.orm import Session

from . import repository as repo
from .ai_flavors import models_in_tier, size_flavor
from .db import get_session
from .engine import SizingError, calculate, size_ml
from .engine.config import EngineSettings
from .engine.ml import CATEGORIES as ML_CATEGORIES
from .exporters import EXPORTERS
from .inventory import StaticOpenStackGPUInventory
from .openstack_compiler import OpenStackFlavorCompiler
from .schemas import (
    AIFlavorDefinition,
    BenchmarkRecord,
    CalculateRequest,
    CalculationResult,
    CompareGPUsRequest,
    CompareModelsRequest,
    ComparePrecisionRequest,
    CustomModelRequest,
    DeploymentSpecCreate,
    DeploymentSpecRecord,
    FlavorSizeRequest,
    FlavorSizeResult,
    GPUSpec,
    InventoryRecord,
    MLSizingRequest,
    MLSizingResult,
    ModelSpec,
    ModelStatusPatch,
    SaveCalculationRequest,
    VGPUProfile,
    YamlImportRequest,
)

router = APIRouter(prefix="/api/v1")


def require_admin(x_admin_token: Optional[str] = Header(default=None)) -> None:
    """Catalog mutations require X-Admin-Token when ADMIN_TOKEN is set."""
    expected = os.getenv("ADMIN_TOKEN")
    if expected and x_admin_token != expected:
        raise HTTPException(status_code=401, detail="Admin token required")


# ---------------------------------------------------------------------------- core calculation


def _resolve_model(session: Session, req: CalculateRequest) -> ModelSpec:
    if req.custom_model is not None:
        return req.custom_model.model_copy(update={"metadata_status": "user_defined"})
    model = repo.get_model(session, req.model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_id}' not found")
    return model


def run_calculation(
    session: Session, req: CalculateRequest, site: Optional[str] = None, with_suggestions: bool = False
) -> CalculationResult:
    model = _resolve_model(session, req)
    try:
        return calculate(req, model, repo.catalog_snapshot(session, site), repo.get_settings(session), with_suggestions)
    except SizingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/calculate", response_model=CalculationResult)
def post_calculate(req: CalculateRequest, site: Optional[str] = None, session: Session = Depends(get_session)):
    return run_calculation(session, req, site, with_suggestions=True)


@router.post("/calculate/export")
def export_calculation(
    req: CalculateRequest,
    format: str = Query("json", pattern="^(json|yaml|csv|markdown)$"),
    session: Session = Depends(get_session),
):
    result = run_calculation(session, req)
    fn, media, ext = EXPORTERS[format]
    filename = f"gpu-sizing-{result.model.id}-{result.workload.precision}.{ext}"
    return Response(
        content=fn(result), media_type=media, headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@router.get("/meta")
def meta(session: Session = Depends(get_session)) -> dict[str, Any]:
    s = repo.get_settings(session)
    return {
        "precisions": [
            {
                "id": k,
                "label": v.label,
                "bytes_per_parameter": v.bytes_per_parameter,
                "default_overhead_factor": v.default_overhead_factor,
            }
            for k, v in s.precision.items()
        ],
        "kv_precisions": [{"id": k, "label": v.label} for k, v in s.kv_precision.items()],
        "frameworks": [{"id": k, "label": v.label} for k, v in s.frameworks.items()],
        "priorities": [
            {
                "id": k,
                "label": v.label,
                "safety_margin_percent": v.safety_margin_percent,
                "tokens_per_second_per_user": v.tokens_per_second_per_user,
            }
            for k, v in s.priorities.items()
        ],
        "environments": [{"id": k, **v.model_dump()} for k, v in s.environments.items()],
        "context_defaults": s.context.model_dump(),
        "gpu_counts": s.gpu_counts,
    }


# ---------------------------------------------------------------------------- comparisons


@router.post("/compare/models")
def compare_models(body: CompareModelsRequest, session: Session = Depends(get_session)) -> list[dict[str, Any]]:
    rows = []
    for model_id in body.model_ids:
        model = repo.get_model(session, model_id)
        if model is None:
            rows.append({"model_id": model_id, "error": "Model not found"})
            continue
        workload = body.workload.model_dump()
        note = None
        if model.max_context_length and workload["context_length"] > model.max_context_length:
            workload["context_length"] = model.max_context_length
            note = f"Context capped at model maximum ({model.max_context_length:,})"
        try:
            res = run_calculation(session, CalculateRequest(model_id=model_id, **workload))
        except HTTPException as exc:
            rows.append({"model_id": model_id, "name": model.name, "error": exc.detail})
            continue
        rows.append(_summary_row(res, note))
    return rows


@router.post("/compare/precisions")
def compare_precisions(body: ComparePrecisionRequest, session: Session = Depends(get_session)) -> list[dict[str, Any]]:
    model = _resolve_model(session, body.request)
    settings = repo.get_settings(session)
    precisions = body.precisions or [p for p in ("int4", "int8", "fp8", "bf16") if p in settings.precision]
    for extra in model.supported_serving_precisions:
        if not body.precisions and extra not in precisions and extra in settings.precision:
            precisions.append(extra)
    rows = []
    for p in precisions:
        req = body.request.model_copy(update={"precision": p})
        try:
            res = run_calculation(session, req)
        except HTTPException as exc:
            rows.append({"precision": p, "error": exc.detail})
            continue
        row = _summary_row(res)
        row["supported"] = not model.supported_serving_precisions or p in model.supported_serving_precisions
        rows.append(row)
    return rows


@router.post("/compare/gpus")
def compare_gpus(body: CompareGPUsRequest, session: Session = Depends(get_session)) -> list[dict[str, Any]]:
    gpus = [g for g in repo.list_gpus(session) if g.enabled]
    if body.gpu_ids:
        gpus = [g for g in gpus if g.id in body.gpu_ids]
    rows = []
    for gpu in gpus:
        req = body.request.model_copy(update={"allowed_gpu_ids": [gpu.id], "allow_vgpu": False})
        res = run_calculation(session, req)
        rec = res.recommendation
        c = rec.candidate if rec else next((x for x in res.rejected if x.gpu_id == gpu.id), None)
        rows.append(
            {
                "gpu_id": gpu.id,
                "gpu": gpu.display_name,
                "vram_gb": gpu.vram_gb,
                "memory_bandwidth_gbps": gpu.memory_bandwidth_gbps,
                "fits": rec is not None,
                "gpu_count": rec.count if rec else None,
                "utilization_percent": c.utilization_percent if c else None,
                "required_per_gpu_gb": c.required_per_unit_gb if c else None,
                "bandwidth_ratio": c.bandwidth_ratio if c else None,
                "compute_ratio": c.compute_ratio if c else None,
                "est_decode_tokens_per_second_per_user": c.est_decode_tokens_per_second_per_user if c else None,
                "cost": c.cost.model_dump() if c and c.cost else None,
                "confidence": c.confidence if c else "estimated",
                "benchmark": c.benchmark.model_dump() if c and c.benchmark else None,
                "reason": None if rec else (c.rejected_reason if c else "Not evaluated"),
                "notes": c.notes if c else [],
            }
        )
    return rows


def _summary_row(res: CalculationResult, note: Optional[str] = None) -> dict[str, Any]:
    rec = res.recommendation
    return {
        "model_id": res.model.id,
        "name": res.model.name,
        "architecture": res.model.architecture,
        "total_parameters_b": res.model.total_parameters_b,
        "active_parameters_b": res.model.active_parameters_b,
        "precision": res.workload.precision,
        "context_length": res.workload.context_length,
        "weights_gb": res.memory.model_weights_gb,
        "kv_cache_gb": res.memory.kv_cache_gb,
        "required_vram_gb": res.memory.required_vram_gb,
        "recommended": rec.summary if rec else None,
        "gpu_count": rec.count if rec else None,
        "mode": rec.mode if rec else None,
        "ai_flavor": rec.ai_flavor if rec else None,
        "cost": rec.candidate.cost.model_dump() if rec and rec.candidate.cost else None,
        "confidence": res.confidence_label,
        "note": note,
    }


# ---------------------------------------------------------------------------- models


@router.get("/models", response_model=list[ModelSpec])
def get_models(include_disabled: bool = False, q: Optional[str] = None, session: Session = Depends(get_session)):
    models = repo.list_models(session, include_disabled)
    if q:
        terms = q.lower().split()
        models = [m for m in models if all(t in f"{m.name} {m.family} {m.vendor} {m.id}".lower() for t in terms)]
    return models


@router.get("/models/export")
def export_models(session: Session = Depends(get_session)):
    return Response(
        content=repo.export_models_yaml(session),
        media_type="application/yaml",
        headers={"Content-Disposition": 'attachment; filename="models.yaml"'},
    )


@router.post("/models/import", dependencies=[Depends(require_admin)])
def import_models(body: YamlImportRequest, session: Session = Depends(get_session)):
    try:
        return repo.import_models_yaml(session, body.yaml, body.overwrite)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/models/{model_id}", response_model=ModelSpec)
def get_model(model_id: str, session: Session = Depends(get_session)):
    model = repo.get_model(session, model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.post("/models", response_model=ModelSpec, status_code=201, dependencies=[Depends(require_admin)])
def create_model(spec: ModelSpec, session: Session = Depends(get_session)):
    if repo.get_model(session, spec.id):
        raise HTTPException(status_code=409, detail="Model id already exists")
    return repo.upsert_model(session, spec)


@router.put("/models/{model_id}", response_model=ModelSpec, dependencies=[Depends(require_admin)])
def update_model(model_id: str, spec: ModelSpec, session: Session = Depends(get_session)):
    if spec.id != model_id:
        raise HTTPException(status_code=400, detail="Path id and body id differ")
    if repo.get_model(session, model_id) is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return repo.upsert_model(session, spec)


@router.patch("/models/{model_id}", response_model=ModelSpec, dependencies=[Depends(require_admin)])
def patch_model(model_id: str, patch: ModelStatusPatch, session: Session = Depends(get_session)):
    model = repo.get_model(session, model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    updated = model.model_copy(update=patch.model_dump(exclude_none=True))
    return repo.upsert_model(session, updated)


@router.post("/custom-model")
def custom_model(body: CustomModelRequest, session: Session = Depends(get_session), x_admin_token: Optional[str] = Header(default=None)):
    """Validate a user-defined model; optionally persist it to the catalog (admin)."""
    spec = body.model.model_copy(update={"metadata_status": "user_defined"})
    if body.persist:
        require_admin(x_admin_token)
        if repo.get_model(session, spec.id):
            raise HTTPException(status_code=409, detail="Model id already exists")
        repo.upsert_model(session, spec)
    return {
        "model": spec,
        "persisted": body.persist,
        "kv_metadata": "exact" if spec.has_exact_kv_metadata else "heuristic",
        "resolved_head_dim": spec.resolved_head_dim,
    }


# ---------------------------------------------------------------------------- GPUs / vGPU / inventory


@router.get("/gpus", response_model=list[GPUSpec])
def get_gpus(session: Session = Depends(get_session)):
    return repo.list_gpus(session)


@router.post("/gpus", response_model=GPUSpec, status_code=201, dependencies=[Depends(require_admin)])
def create_gpu(spec: GPUSpec, session: Session = Depends(get_session)):
    if repo.get_gpu(session, spec.id):
        raise HTTPException(status_code=409, detail="GPU id already exists")
    return repo.upsert_gpu(session, spec)


@router.put("/gpus/{gpu_id}", response_model=GPUSpec, dependencies=[Depends(require_admin)])
def update_gpu(gpu_id: str, spec: GPUSpec, session: Session = Depends(get_session)):
    if spec.id != gpu_id:
        raise HTTPException(status_code=400, detail="Path id and body id differ")
    return repo.upsert_gpu(session, spec)


@router.get("/vgpu-profiles", response_model=list[VGPUProfile])
def get_vgpu_profiles(session: Session = Depends(get_session)):
    return repo.list_profiles(session)


@router.post("/vgpu-profiles", response_model=VGPUProfile, dependencies=[Depends(require_admin)])
def upsert_vgpu_profile(spec: VGPUProfile, session: Session = Depends(get_session)):
    if repo.get_gpu(session, spec.physical_gpu) is None:
        raise HTTPException(status_code=422, detail=f"Unknown physical GPU '{spec.physical_gpu}'")
    return repo.upsert_profile(session, spec)


@router.get("/inventory", response_model=list[InventoryRecord])
def get_inventory(session: Session = Depends(get_session)):
    return repo.list_inventory(session)


@router.put("/inventory", response_model=list[InventoryRecord], dependencies=[Depends(require_admin)])
def put_inventory(records: list[InventoryRecord], session: Session = Depends(get_session)):
    known = {g.id for g in repo.list_gpus(session)}
    unknown = sorted({r.gpu_id for r in records} - known)
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown GPU ids: {', '.join(unknown)}")
    return repo.replace_inventory(session, records)


# ---------------------------------------------------------------------------- benchmarks


@router.get("/benchmarks", response_model=list[BenchmarkRecord])
def get_benchmarks(model_id: Optional[str] = None, gpu_id: Optional[str] = None, session: Session = Depends(get_session)):
    return repo.list_benchmarks(session, model_id, gpu_id)


@router.post("/benchmarks", response_model=BenchmarkRecord, status_code=201, dependencies=[Depends(require_admin)])
def create_benchmark(rec: BenchmarkRecord, session: Session = Depends(get_session)):
    if repo.get_model(session, rec.model_id) is None:
        raise HTTPException(status_code=422, detail=f"Unknown model '{rec.model_id}'")
    if repo.get_gpu(session, rec.gpu_id) is None:
        raise HTTPException(status_code=422, detail=f"Unknown GPU '{rec.gpu_id}'")
    return repo.add_benchmark(session, rec)


@router.delete("/benchmarks/{bench_id}", status_code=204, dependencies=[Depends(require_admin)])
def remove_benchmark(bench_id: int, session: Session = Depends(get_session)):
    if not repo.delete_benchmark(session, bench_id):
        raise HTTPException(status_code=404, detail="Benchmark not found")
    return Response(status_code=204)


# ---------------------------------------------------------------------------- saved calculations


@router.get("/calculations")
def get_calculations(session: Session = Depends(get_session)) -> list[dict[str, Any]]:
    out = []
    for row in repo.list_calculations(session):
        rec = (row.result or {}).get("recommendation") or {}
        out.append(
            {
                "id": row.id,
                "name": row.name,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "model": (row.result or {}).get("model", {}).get("name"),
                "precision": row.request.get("precision"),
                "required_vram_gb": (row.result or {}).get("memory", {}).get("required_vram_gb"),
                "recommended": rec.get("summary"),
                "ai_flavor": rec.get("ai_flavor"),
            }
        )
    return out


@router.post("/calculations", status_code=201)
def create_calculation(body: SaveCalculationRequest, session: Session = Depends(get_session)) -> dict[str, Any]:
    result = run_calculation(session, body.request)
    row = repo.save_calculation(
        session, body.name, body.request.model_dump(mode="json", exclude_none=True), result.model_dump(mode="json")
    )
    return {"id": row.id, "name": row.name}


@router.get("/calculations/{calc_id}")
def get_calculation(calc_id: int, session: Session = Depends(get_session)) -> dict[str, Any]:
    row = repo.get_calculation(session, calc_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Calculation not found")
    return {"id": row.id, "name": row.name, "created_at": row.created_at.isoformat(), "request": row.request, "result": row.result}


@router.delete("/calculations/{calc_id}", status_code=204)
def remove_calculation(calc_id: int, session: Session = Depends(get_session)):
    if not repo.delete_calculation(session, calc_id):
        raise HTTPException(status_code=404, detail="Calculation not found")
    return Response(status_code=204)


# ---------------------------------------------------------------------------- AI VM flavors


@router.get("/ai-flavors")
def get_ai_flavors(include_disabled: bool = False, session: Session = Depends(get_session)) -> list[dict[str, Any]]:
    """Flavor tiers with the catalog models that fall into each tier."""
    tiers = repo.list_flavors(session, include_disabled=include_disabled)
    models = repo.list_models(session)
    settings = repo.get_settings(session)
    enabled = [t for t in tiers if t.enabled]
    return [
        {**t.model_dump(mode="json"), "model_ids": models_in_tier(t, models, enabled, settings) if t.enabled else []}
        for t in tiers
    ]


@router.post("/ai-flavors/size", response_model=FlavorSizeResult)
def post_size_flavor(body: FlavorSizeRequest, session: Session = Depends(get_session)):
    if body.flavor_id and repo.get_flavor(session, body.flavor_id) is None:
        raise HTTPException(status_code=404, detail=f"AI flavor '{body.flavor_id}' not found")
    if body.custom_model is not None:
        model = body.custom_model.model_copy(update={"metadata_status": "user_defined"})
    else:
        model = repo.get_model(session, body.model_id)
        if model is None:
            raise HTTPException(status_code=404, detail=f"Model '{body.model_id}' not found")
    try:
        return size_flavor(body, model, repo.catalog_snapshot(session), repo.get_settings(session))
    except SizingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/ai-flavors/{flavor_id}", response_model=AIFlavorDefinition)
def get_ai_flavor(flavor_id: str, session: Session = Depends(get_session)):
    flavor = repo.get_flavor(session, flavor_id)
    if flavor is None:
        raise HTTPException(status_code=404, detail="AI flavor not found")
    return flavor


@router.post("/ai-flavors", response_model=AIFlavorDefinition, status_code=201, dependencies=[Depends(require_admin)])
def create_ai_flavor(spec: AIFlavorDefinition, session: Session = Depends(get_session)):
    if repo.get_flavor(session, spec.id):
        raise HTTPException(status_code=409, detail="AI flavor id already exists")
    return repo.upsert_flavor(session, spec)


@router.put("/ai-flavors/{flavor_id}", response_model=AIFlavorDefinition, dependencies=[Depends(require_admin)])
def update_ai_flavor(flavor_id: str, spec: AIFlavorDefinition, session: Session = Depends(get_session)):
    if spec.id != flavor_id:
        raise HTTPException(status_code=400, detail="Path id and body id differ")
    if repo.get_flavor(session, flavor_id) is None:
        raise HTTPException(status_code=404, detail="AI flavor not found")
    return repo.upsert_flavor(session, spec)


@router.delete("/ai-flavors/{flavor_id}", status_code=204, dependencies=[Depends(require_admin)])
def remove_ai_flavor(flavor_id: str, session: Session = Depends(get_session)):
    if not repo.delete_flavor(session, flavor_id):
        raise HTTPException(status_code=404, detail="AI flavor not found")
    return Response(status_code=204)


# ---------------------------------------------------------------------------- deployment specs


def _spec_record(row) -> DeploymentSpecRecord:
    return DeploymentSpecRecord(
        id=row.id,
        name=row.name,
        created_at=row.created_at.isoformat() if row.created_at else "",
        workload_type=row.workload_type,
        ai_flavor=row.ai_flavor,
        spec=row.spec,
        openstack=row.openstack,
    )


@router.post("/deployment-specs", response_model=DeploymentSpecRecord, status_code=201)
def create_deployment_spec(body: DeploymentSpecCreate, session: Session = Depends(get_session)):
    """"Deploy" for the MVP: store the AIFlavorDeploymentSpec / MLDeploymentSpec and compile the
    OpenStack flavor recommendation. Nothing is provisioned."""
    try:
        compiled = OpenStackFlavorCompiler(StaticOpenStackGPUInventory(session)).compile(body.spec)
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid deployment spec: {exc}") from exc
    return _spec_record(repo.save_deployment_spec(session, body.name, body.spec, compiled))


@router.get("/deployment-specs", response_model=list[DeploymentSpecRecord])
def get_deployment_specs(session: Session = Depends(get_session)):
    return [_spec_record(r) for r in repo.list_deployment_specs(session)]


@router.delete("/deployment-specs/{spec_id}", status_code=204)
def remove_deployment_spec(spec_id: int, session: Session = Depends(get_session)):
    if not repo.delete_deployment_spec(session, spec_id):
        raise HTTPException(status_code=404, detail="Deployment spec not found")
    return Response(status_code=204)


# ---------------------------------------------------------------------------- custom ML models


@router.get("/ml/meta")
def ml_meta(session: Session = Depends(get_session)) -> dict[str, Any]:
    s = repo.get_settings(session)
    return {
        "categories": [
            {"id": k, "label": v.label, "input_kind": v.input_kind, "defaults": v.defaults} for k, v in ML_CATEGORIES.items()
        ],
        "frameworks": [{"id": k, "label": v.label, "nvidia_only": v.nvidia_only} for k, v in s.ml.frameworks.items()],
        "optimizers": [{"id": k, "label": v.label} for k, v in s.ml.optimizers.items()],
        "precisions": [
            {"id": k, "label": v.label, "bytes_per_parameter": v.bytes_per_parameter}
            for k, v in s.precision.items()
            if k in ("fp32", "tf32", "fp16", "bf16", "fp8", "int8", "int4")
        ],
        "default_adapter_percent": s.ml.default_adapter_percent,
    }


@router.post("/ml/size", response_model=MLSizingResult)
def post_ml_size(body: MLSizingRequest, session: Session = Depends(get_session)):
    try:
        return size_ml(body, repo.catalog_snapshot(session), repo.get_settings(session), with_suggestions=True)
    except SizingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ---------------------------------------------------------------------------- settings


@router.get("/settings", response_model=EngineSettings)
def get_settings(session: Session = Depends(get_session)):
    return repo.get_settings(session)


@router.put("/settings", response_model=EngineSettings, dependencies=[Depends(require_admin)])
def put_settings(value: dict[str, Any], session: Session = Depends(get_session)):
    try:
        return repo.put_settings(session, value)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors(include_url=False)) from exc


@router.post("/settings/reset", response_model=EngineSettings, dependencies=[Depends(require_admin)])
def reset_settings(session: Session = Depends(get_session)):
    return repo.reset_settings(session)
