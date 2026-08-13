"""Wrapper around ../split-llm-simple/split_llm.py.

Flow:
  1. run_split_llm(model)              — run split_llm.py with defaults via its own venv; cache by model.
  2. copy_splits_to_work_dir(...)      — copy into the build context; returns (safetensors, metadata).

Size-derivation and packing live in shared.packing (compute_split_stats, repack,
layers_for_percent) — they are weight-format agnostic and shared with the CV arm.
"""

import json
import os
import shutil
import subprocess

from shared import fs, log
from shared.config import load_config

_SPLIT_REPO = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "split-llm-simple")
)


def split_llm_slug(model: str) -> str:
    """Matches the slug used by split_llm.py for its output folder."""
    return model.replace("/", "_").replace(":", "_")


# ── 1. invoke splitter ─────────────────────────────────────────────────


def run_split_llm(model: str) -> str:
    """Run ../split-llm-simple/split_llm.py with default args, using that
    repo's own .venv. Caches via manifest.json: re-runs only if the model
    changed (defaults are fixed).

    Returns the splits_output/<slug>/ path.
    """
    slug = split_llm_slug(model)
    out_dir = os.path.join(_SPLIT_REPO, "splits_output", slug)
    manifest_path = os.path.join(out_dir, "manifest.json")

    if os.path.exists(manifest_path):
        try:
            with open(manifest_path) as f:
                m = json.load(f)
            if m.get("model_name") == model:
                log.info(f"Splits cache hit for {model} at {out_dir}")
                return out_dir
            log.info(
                f"Splits cache stale for {model} "
                f"(was {m.get('model_name')}) — regenerating"
            )
        except (OSError, json.JSONDecodeError):
            log.info(f"Splits manifest unreadable at {manifest_path} — regenerating")
        fs.rmtree(out_dir)

    venv_py = os.path.join(_SPLIT_REPO, ".venv-split-llm/bin/python")
    script = os.path.join(_SPLIT_REPO, "split_llm.py")
    if not os.path.exists(venv_py):
        raise RuntimeError(f"split-llm-simple venv not found at {venv_py}")

    cmd = [venv_py, script, "--model", model]
    env = os.environ.copy()
    hf_home = load_config().hf_home
    if hf_home:
        env["HF_HOME"] = hf_home
        log.info(f"Setting HF_HOME={hf_home}")
    log.info(f"Running split_llm: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=_SPLIT_REPO, check=True, env=env)
    log.result(f"Splits generated at {out_dir}")
    return out_dir


# ── 2. copy into build context ─────────────────────────────────────────


def split_metadata_paths(work_dir: str) -> list[str]:
    """Sorted absolute paths of model-metadata files (config,
    tokenizer, generation config, ...) in work_dir.
    """
    return sorted(
        os.path.join(work_dir, f)
        for f in os.listdir(work_dir)
        if not f.endswith(".safetensors") and f != "manifest.json"
    )


def copy_splits_to_work_dir(splits_dir: str, work_dir: str) -> tuple[list[str], list[str]]:
    """Copy .safetensors + model metadata (config/tokenizer/...) from splits_dir
    into work_dir so that 2dfs.json and Dockerfile COPY paths stay inside the
    build context. Skips copy when work_dir/manifest.json already matches.

    Returns (safetensor_paths, metadata_paths): the .safetensors weights and the
    metadata files in work_dir, both sorted. Callers pack both into the image —
    metadata is typically pinned to allotment 0.
    """
    os.makedirs(work_dir, exist_ok=True)

    src_manifest = os.path.join(splits_dir, "manifest.json")
    dst_manifest = os.path.join(work_dir, "manifest.json")
    if (os.path.exists(dst_manifest) and os.path.exists(src_manifest)
            and _read(src_manifest) == _read(dst_manifest)):
        log.info(f"Chunks up-to-date at {work_dir} — skipping copy")
    else:
        fs.clear_dir(work_dir)
        for f in sorted(os.listdir(splits_dir)):
            # field.json describes split_llm's own bucketing, which we ignore.
            if f == "field.json":
                continue
            src = os.path.join(splits_dir, f)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(work_dir, f))
        log.result(f"Copied splits into {work_dir}")

    safetensor_paths = sorted(
        os.path.join(work_dir, f)
        for f in os.listdir(work_dir)
        if f.endswith(".safetensors")
    )
    return safetensor_paths, split_metadata_paths(work_dir)


def _read(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()
