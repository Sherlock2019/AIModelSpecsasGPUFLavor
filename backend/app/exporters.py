"""Export a calculation as JSON, YAML, CSV or a Markdown report."""

from __future__ import annotations

import csv
import io
import json
from typing import Any

import yaml

from .schemas import CalculationResult


def to_json(result: CalculationResult) -> str:
    return json.dumps(result.model_dump(mode="json"), indent=2)


def to_yaml(result: CalculationResult) -> str:
    return yaml.safe_dump(result.model_dump(mode="json"), sort_keys=False, allow_unicode=True)


def _flatten(prefix: str, value: Any, out: list[tuple[str, Any]]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}" if prefix else k, v, out)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _flatten(f"{prefix}[{i}]", v, out)
    else:
        out.append((prefix, value))


def to_csv(result: CalculationResult) -> str:
    data = result.model_dump(mode="json", exclude={"openstack_yaml"})
    rows: list[tuple[str, Any]] = []
    for section in ("calculation_mode", "model", "workload", "weights", "kv_cache", "memory", "compute", "recommendation", "alternatives", "warnings", "trace"):
        _flatten(section, data.get(section), rows)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["field", "value"])
    writer.writerows(rows)
    return buf.getvalue()


def to_markdown(result: CalculationResult) -> str:
    m, wl, mem, c = result.model, result.workload, result.memory, result.compute
    rec = result.recommendation
    lines = [
        f"# GPU sizing report — {m.name} / {wl.precision.upper()}",
        "",
        f"**Confidence:** {result.confidence_label}  ",
        f"**Model metadata:** {m.metadata_status.upper().replace('_', '-')}"
        + (f" ([source]({m.source_url}))" if m.source_url else ""),
        "",
        "## Summary",
        "",
        f"- Estimated VRAM required: **{mem.required_vram_gb:.1f} GB**",
    ]
    if rec:
        lines += [
            f"- Recommended: **{rec.summary}**",
            f"- AI flavor: **{rec.ai_flavor}** (`{rec.flavor_name}`)",
            f"- VRAM utilization: {rec.utilization_percent:.0f}% · headroom {rec.headroom_gb:.1f} GB",
        ]
    else:
        lines.append("- **No configuration fits.**")
    if result.alternatives:
        lines.append(f"- Alternatives: {'; '.join(a.label for a in result.alternatives)}")
    lines.append(f"- vGPU: {result.vgpu_verdict}")
    lines += [
        "",
        "## Workload",
        "",
        "| Input | Value |",
        "|---|---|",
        f"| Architecture | {m.architecture} ({m.total_parameters_b:g}B total / {m.active_parameters_b:g}B active) |",
        f"| Serving precision | {wl.precision.upper()} (KV cache {wl.kv_cache_precision.upper()}) |",
        f"| Context | {wl.context_length:,} tokens ({wl.context_mode} mode) |",
        f"| Concurrent sequences | {wl.concurrent_sequences} |",
        f"| Avg input / output tokens | {wl.average_input_tokens:,} / {wl.average_output_tokens:,} |",
        f"| Priority / framework | {wl.performance_priority} / {wl.framework} |",
        "",
        "## Memory breakdown",
        "",
        "| Component | GB |",
        "|---|---:|",
        f"| Model weights | {mem.model_weights_gb:.2f} |",
        f"| KV cache | {mem.kv_cache_gb:.2f} |",
        f"| Runtime overhead | {mem.runtime_overhead_gb:.2f} |",
        f"| Workspace | {mem.workspace_gb:.2f} |",
        f"| Communication | {mem.communication_gb:.2f} |",
        f"| Subtotal | {mem.subtotal_gb:.2f} |",
        f"| Safety headroom ({mem.safety_margin_percent:g}%) | {mem.headroom_gb:.2f} |",
        f"| **Total required** | **{mem.required_vram_gb:.2f}** |",
        "",
        "## Compute profile",
        "",
        f"- Compute class: {c.classification.replace('_', ' ').upper()} (score {c.compute_demand_score:,.0f})",
        f"- Memory intensity: {c.memory_intensity.replace('_', ' ').upper()}",
        f"- Requested decode throughput: {c.requested_tokens_per_second:,.0f} tokens/s "
        f"({c.tokens_per_second_per_user:g}/user)",
        "",
        "## Why this recommendation",
        "",
    ]
    lines += [f"{i}. {line}" for i, line in enumerate(result.explanation, 1)]
    if result.warnings:
        lines += ["", "## Warnings", ""]
        lines += [f"- **{w.severity.upper()}** — {w.message}" for w in result.warnings]
    lines += ["", "## Calculation trace", "", "| Step | Formula | Value |", "|---|---|---|"]
    lines += [f"| {t.step} | {t.formula} | {t.value} |" for t in result.trace]
    if result.openstack_yaml:
        lines += ["", "## OpenStack flavor (recommendation only — not deployed)", "", "```yaml", result.openstack_yaml.rstrip(), "```"]
    lines += ["", "_All performance figures are estimates unless labelled BENCHMARK-BASED._", ""]
    return "\n".join(lines)


EXPORTERS = {
    "json": (to_json, "application/json", "json"),
    "yaml": (to_yaml, "application/yaml", "yaml"),
    "csv": (to_csv, "text/csv", "csv"),
    "markdown": (to_markdown, "text/markdown", "md"),
}
