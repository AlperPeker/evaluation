import os

from shared import fs, log, paths
from shared.artifacts import (
    create_base_dockerfile, create_stargz_dockerfile, write_2dfs_json,
)
from shared.config import EnvConfig
from shared.registry import plain_base_image, stargz_base_image
from shared.packing import compute_split_stats, repack
from shared.split_llm import (
    copy_splits_to_work_dir, run_split_llm, split_metadata_paths,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def clear_chunks(model_name: str | None = None) -> None:
    if model_name is None:
        fs.clear_dir(paths.chunks_dir(SCRIPT_DIR))
    else:
        fs.clear_dir(paths.model_chunks_dir(SCRIPT_DIR, model_name))


def prepare_model_splits(model_name: str) -> tuple[str, int]:
    """Ensure splits exist on disk and return (chunks_dir, max_allotments)."""
    chunks_dir = paths.model_chunks_dir(SCRIPT_DIR, model_name)
    safetensor_paths = sorted(
        os.path.join(chunks_dir, f)
        for f in os.listdir(chunks_dir)
        if f.endswith(".safetensors")
    ) if os.path.isdir(chunks_dir) else []
    if safetensor_paths:
        log.info(f"Chunks already present at {chunks_dir} — skipping split_llm")
    else:
        splits_dir = run_split_llm(model_name)
        safetensor_paths, _ = copy_splits_to_work_dir(splits_dir, chunks_dir)
    max_allotments, _, _ = compute_split_stats(safetensor_paths)
    return chunks_dir, max_allotments


def generate_build_artifacts(
    chunks_dir: str, num_layers: int,
    source_image: str = "", cfg: EnvConfig = None,
) -> list[list[str]]:
    """Produce 2dfs.json + Dockerfiles for one (chunks_dir, capacity) build.

    Returns the groups (one list of file paths per allotment/bucket), in the
    same order as the COPY layers in the resulting Dockerfile and the
    allotments in 2dfs.json. groups[0] is the bottommost image layer,
    groups[-1] is the topmost.
    """
    files = sorted(os.path.join(chunks_dir, f) for f in os.listdir(chunks_dir))
    safetensor_paths = [p for p in files if p.endswith(".safetensors")]
    metadata_files = split_metadata_paths(chunks_dir)

    groups = repack(safetensor_paths, num_layers, extra_files=metadata_files)

    write_2dfs_json(groups, SCRIPT_DIR)
    create_stargz_dockerfile(groups, stargz_base_image(source_image, cfg), SCRIPT_DIR)
    create_base_dockerfile(groups, plain_base_image(source_image, cfg), SCRIPT_DIR)
    return groups


def preview_packings(
    chunks_dir: str, num_layers_list: list[int],
) -> list[list[list[str]]]:
    """Repack the already-prepared chunks into N buckets for each N in num_layers_list."""
    files = sorted(os.path.join(chunks_dir, f) for f in os.listdir(chunks_dir))
    safetensor_paths = [p for p in files if p.endswith(".safetensors")]
    if not safetensor_paths:
        raise RuntimeError(
            f"No .safetensors in {chunks_dir} — call generate_build_artifacts() before "
            f"preview_packings()."
        )

    metadata_files = split_metadata_paths(chunks_dir)

    n_available = len(safetensor_paths)
    results: list[list[list[str]]] = []
    for n in num_layers_list:
        n_eff = min(max(1, n), n_available)
        groups = repack(safetensor_paths, n_eff, extra_files=metadata_files)
        results.append(groups)
    return results


def split_stats(chunks_dir: str) -> dict:
    """Aggregate split-pool stats (sizes in MB) for run metadata."""
    files = sorted(os.path.join(chunks_dir, f) for f in os.listdir(chunks_dir))
    safetensor_paths = [p for p in files if p.endswith(".safetensors")]
    _, total_size, max_file_size = compute_split_stats(safetensor_paths)
    return {
        "num_safetensors": len(safetensor_paths),
        "total_mb": round(total_size / (1024 ** 2), 1),
        "max_file_mb": round(max_file_size / (1024 ** 2), 1),
    }


def packing_preview_data(
    chunks_dir: str, labels: list[str], num_layers_list: list[int],
    capacities: list[int] | None = None,
) -> list[dict]:
    """Structured per-label packing: #allotments + allotment sizes (MB)."""
    packings = preview_packings(chunks_dir, num_layers_list)
    out: list[dict] = []
    for i, (label, groups) in enumerate(zip(labels, packings)):
        sizes_mb = [
            round(sum(os.path.getsize(p) for p in g) / (1024 ** 2), 1) for g in groups
        ]
        entry: dict = {}
        if capacities is not None:
            entry["capacity"] = capacities[i]
        entry.update({
            "label": label,
            "num_layers": len(groups),
            "allotment_sizes_mb": sizes_mb,
        })
        out.append(entry)
    return out


def print_packing_table(
    chunks_dir: str, model_name: str, max_allowed_splits: int,
    labels: list[str], num_layers_list: list[int],
) -> None:
    """Print per-label (#allotments, allotment sizes in MB)."""
    preview = packing_preview_data(chunks_dir, labels, num_layers_list)

    log.result(
        f"\n=== Packing preview: {model_name} "
        f"(max_allowed_splits={max_allowed_splits}) ==="
    )
    log.result(f"{'label':>10}  {'allotments':>11}  sizes (MB)")
    log.result("-" * 60)
    for entry in preview:
        sizes_str = "[" + ", ".join(f"{s:.1f}" for s in entry["allotment_sizes_mb"]) + "]"
        log.result(f"{entry['label']:>10}  {entry['num_layers']:>11}  {sizes_str}")
