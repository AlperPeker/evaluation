import os
from datetime import datetime, timezone

def _model_slug(model: str) -> str:
    return model.replace("/", "--")


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ── directory helpers ──────────────────────────────────────────────

def build_perf_dir(benchmark_dir: str) -> str:
    return os.path.join(benchmark_dir, "build_performance")

def pull_perf_dir(benchmark_dir: str) -> str:
    return os.path.join(benchmark_dir, "pull_performance")

def chunks_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "chunks")

def model_chunks_dir(base_dir: str, model: str) -> str:
    return os.path.join(base_dir, "chunks", _model_slug(model))

def models_dir(base_dir: str, model: str) -> str:
    return os.path.join(base_dir, "models", _model_slug(model))


# ── artifact paths ─────────────────────────────────────────────────

def tdfs_json_path(base_dir: str) -> str:
    return os.path.join(base_dir, "2dfs.json")

def stargz_dockerfile_path(base_dir: str) -> str:
    return os.path.join(base_dir, "Dockerfile.stargz")

def base_dockerfile_path(base_dir: str) -> str:
    return os.path.join(base_dir, "Dockerfile.base")
