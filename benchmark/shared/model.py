import os

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, list_repo_files

load_dotenv()

from shared import fs, log, paths
from shared.registry import clear_registry
from shared.services import clear_2dfs_cache, clear_stargz_cache, prune_buildkit

BUFFER_SIZE = 8 * 1024 * 1024  # 8 MB


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "--")


def download_model(model_name: str, work_dir: str) -> list[str]:
    local_dir = paths.models_dir(work_dir, model_name)
    os.makedirs(local_dir, exist_ok=True)

    # Check if already downloaded — avoids hitting the HF API on every call
    cached = sorted(
        f for f in os.listdir(local_dir)
        if f.endswith(".safetensors") or ("pytorch_model" in f and f.endswith(".bin"))
    )
    if cached:
        log.info(f"Using cached model files in {local_dir}")
        return [os.path.join(local_dir, f) for f in cached]

    token = os.environ.get("HF_TOKEN")
    files = list_repo_files(model_name, token=token)
    shards = sorted(f for f in files if f.endswith(".safetensors"))
    if not shards:
        shards = sorted(f for f in files if f.endswith(".bin") and "pytorch_model" in f)

    local_paths = []
    for filename in shards:
        dest = os.path.join(local_dir, os.path.basename(filename))
        if os.path.exists(dest):
            log.info(f"Skipping {filename} (already exists)")
        else:
            log.info(f"Downloading {filename}...")
            hf_hub_download(
                repo_id=model_name,
                filename=filename,
                token=token,
                local_dir=local_dir,
            )
        local_paths.append(dest)
    return local_paths


def cleanup_pull_experiment(model_name: str, work_dir: str, cfg) -> None:
    """Delete model files, chunks, and 2dfs/buildkit caches after an experiment."""
    log.info(f"Cleaning up experiment artifacts for {model_name}...")
    fs.rmtree(paths.models_dir(work_dir, model_name))
    fs.rmtree(paths.model_chunks_dir(work_dir, model_name))
    clear_2dfs_cache(cfg)
    prune_buildkit()
    clear_stargz_cache()


def cleanup_build_artifacts(build_dir: str, cfg) -> None:
    """Clear build_performance models/chunks and all caches between build and pull phases.

    Preserves base images in the registry (-plain/-esgz/-zstd) so pull scripts can
    reuse them without re-converting.
    """
    log.info("Cleaning up build artifacts...")
    fs.clear_dir(os.path.join(build_dir, "models"))
    fs.clear_dir(paths.chunks_dir(build_dir))
    prune_buildkit()
    clear_registry(cfg, preserve_base=True)
    clear_stargz_cache()


def cleanup_pull_artifacts(experiments: list, pull_dir: str, cfg) -> None:
    """Full cleanup after all pull experiments: per-model cleanup then wipe registry.

    Called at the very end of a nightly run to leave the system in a clean state.
    """
    for model, _ in experiments:
        cleanup_pull_experiment(model, pull_dir, cfg)
    clear_registry(cfg, preserve_base=False)


def split_model(shard_paths: list[str], num_splits: int, work_dir: str, output_dir: str = None) -> list[str]:
    chunk_dir = output_dir if output_dir is not None else os.path.join(work_dir, "chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    chunk_paths = [os.path.join(chunk_dir, f"chunk{i + 1}.bin") for i in range(num_splits)]

    total_size = sum(os.path.getsize(p) for p in shard_paths)
    chunk_size = total_size // num_splits
    log.info(f"Splitting {total_size / 1e9:.2f} GB into {num_splits} chunks (~{chunk_size / 1e9:.2f} GB each)...")

    chunk_idx = 0
    bytes_written = 0
    out = open(chunk_paths[chunk_idx], "wb")

    for shard_path in shard_paths:
        with open(shard_path, "rb") as f:
            while True:
                if chunk_idx < num_splits - 1:
                    to_read = min(BUFFER_SIZE, chunk_size - bytes_written)
                else:
                    to_read = BUFFER_SIZE

                data = f.read(to_read)
                if not data:
                    break

                out.write(data)
                bytes_written += len(data)

                if chunk_idx < num_splits - 1 and bytes_written >= chunk_size:
                    out.close()
                    chunk_idx += 1
                    bytes_written = 0
                    out = open(chunk_paths[chunk_idx], "wb")

    out.close()
    log.info("Split complete.")
    return chunk_paths
