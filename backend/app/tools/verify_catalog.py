"""Check catalog architecture fields against each model's official Hugging Face config.json.

    python -m app.tools.verify_catalog            # report only
    python -m app.tools.verify_catalog --promote  # mark fully-matching models as `verified`

Needs network access. Gated repos (Llama, Gemma) need HF_TOKEN in the environment.
Only JSON metadata is downloaded - nothing is executed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml

MODELS_YAML = Path(__file__).resolve().parent.parent / "data" / "models.yaml"

# catalog field -> candidate config.json keys (first present wins; text_config is searched too)
FIELD_MAP = {
    "hidden_size": ["hidden_size", "d_model"],
    "num_layers": ["num_hidden_layers", "n_layer", "num_layers"],
    "attention_heads": ["num_attention_heads", "n_head"],
    "kv_heads": ["num_key_value_heads", "multi_query_group_num"],
    "head_dim": ["head_dim"],
    "kv_lora_rank": ["kv_lora_rank"],
    "qk_rope_head_dim": ["qk_rope_head_dim"],
    "num_experts": ["num_local_experts", "num_experts", "n_routed_experts"],
    "experts_per_token": ["num_experts_per_tok", "num_experts_per_token", "moe_topk"],
}


def fetch_config(repo: str) -> dict:
    url = f"https://huggingface.co/{repo}/resolve/main/config.json"
    req = urllib.request.Request(url)
    token = os.getenv("HF_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - fixed https host
        return json.loads(resp.read())


def lookup(cfg: dict, keys: list[str]):
    for scope in (cfg, cfg.get("text_config") or {}):
        for k in keys:
            if k in scope and scope[k] is not None:
                return scope[k]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--promote", action="store_true", help="write metadata_status: verified for matching models")
    args = parser.parse_args()

    doc = yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8"))
    changed = False
    for model in doc["models"]:
        src = model.get("source_url") or ""
        prefix = "https://huggingface.co/"
        repo = src[len(prefix):] if src.startswith(prefix) else ""
        if repo.count("/") != 1:
            print(f"SKIP   {model['id']}: no single-repo Hugging Face source_url")
            continue
        try:
            cfg = fetch_config(repo)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"ERROR  {model['id']}: {exc}")
            continue
        mismatches, checked = [], 0
        for field, keys in FIELD_MAP.items():
            if field not in model:
                continue
            upstream = lookup(cfg, keys)
            if upstream is None:
                continue
            checked += 1
            if int(upstream) != int(model[field]):
                mismatches.append(f"{field}: catalog={model[field]} config={upstream}")
        if mismatches:
            print(f"DIFF   {model['id']}: " + "; ".join(mismatches))
        elif checked:
            print(f"OK     {model['id']} ({checked} fields)")
            if args.promote and model.get("metadata_status") != "verified":
                model["metadata_status"] = "verified"
                changed = True
        else:
            print(f"SKIP   {model['id']}: no comparable fields")
    if changed:
        MODELS_YAML.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"Updated {MODELS_YAML}. Re-import it via the Models admin page to update an existing database.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
