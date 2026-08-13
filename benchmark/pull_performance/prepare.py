import json
import os
import subprocess

from shared import log
from shared import paths
from shared.config import EnvConfig
from shared.model import download_model, split_model
from shared.artifacts import write_2dfs_json, create_stargz_dockerfile, create_base_dockerfile, snapshot_artifacts
from shared.registry import stargz_base_image, plain_base_image, zstd_base_image, tdfs_cmd
from shared.services import clear_2dfs_cache
from shared.packing import compute_split_stats, repack
from shared.split_llm import copy_splits_to_work_dir, run_split_llm, split_metadata_paths
from pull_performance.images import (
    build_name_2dfs, build_name_2dfs_stargz, build_name_2dfs_stargz_zstd,
    build_name_stargz, build_name_base,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── chunks ─────────────────────────────────────────────────────────


def prepare_chunks(model_name: str, num_splits: int) -> list[str]:
    chunk_dir = paths.model_chunks_dir(SCRIPT_DIR, model_name)
    chunk_paths = [os.path.join(chunk_dir, f"chunk{i+1}.bin") for i in range(num_splits)]
    if all(os.path.exists(p) for p in chunk_paths):
        log.info("Chunks already exist, skipping download and split.")
        return chunk_paths
    os.makedirs(chunk_dir, exist_ok=True)
    shard_paths = download_model(model_name, SCRIPT_DIR)
    return split_model(shard_paths, num_splits, SCRIPT_DIR, output_dir=chunk_dir)


# ── safetensors splits ──────────────────────────────────────────────


def prepare_model_splits(
    model_name: str, max_allotments_cap: int | None = None,
) -> tuple[list[list[str]], int]:
    """Run split_llm, copy splits into the build context, repack into allotments.

    Returns (allotments, max_allotments) where allotments is the best-fit pack
    of safetensors files into max_allotments buckets. If max_allotments_cap is
    given and smaller, uses that many buckets instead.
    """
    chunks_dir = paths.model_chunks_dir(SCRIPT_DIR, model_name)
    chunks_manifest = os.path.join(chunks_dir, "manifest.json")

    if os.path.exists(chunks_manifest):
        try:
            with open(chunks_manifest) as f:
                m = json.load(f)
            if m.get("model_name") == model_name:
                log.info(f"Chunks cache hit for {model_name} at {chunks_dir} — skipping split_llm and copy")
                safetensor_paths = sorted(
                    os.path.join(chunks_dir, f)
                    for f in os.listdir(chunks_dir)
                    if f.endswith(".safetensors")
                )
                metadata_files = split_metadata_paths(chunks_dir)
                n = _capped_allotments(safetensor_paths, max_allotments_cap)
                return repack(safetensor_paths, n, extra_files=metadata_files), n
        except (OSError, json.JSONDecodeError):
            pass

    splits_dir = run_split_llm(model_name)
    safetensor_paths, metadata_files = copy_splits_to_work_dir(splits_dir, chunks_dir)
    n = _capped_allotments(safetensor_paths, max_allotments_cap)
    return repack(safetensor_paths, n, extra_files=metadata_files), n


def _capped_allotments(safetensor_paths: list[str], cap: int | None) -> int:
    max_allotments, _, _ = compute_split_stats(safetensor_paths)
    if cap is not None and cap < max_allotments:
        log.info(f"Capping allotments {max_allotments} -> {cap}")
        return cap
    return max_allotments


# ── build + push per mode (allotments-native) ───────────────────────


def build_and_push_2dfs_image(
    allotments: list[list[str]],
    cfg: EnvConfig,
    target: str,
    base_image: str,
    extra_flags: list[str],
    label: str,
    artifacts_dir: str | None = None,
) -> None:
    write_2dfs_json(allotments, SCRIPT_DIR)
    if artifacts_dir:
        snapshot_artifacts(SCRIPT_DIR, artifacts_dir)
    cmd = tdfs_cmd(cfg, SCRIPT_DIR) + [
        "build",
        "--platforms", "linux/amd64",
        *extra_flags,
        "--force-http",
        "-f", "2dfs.json",
        base_image,
        target,
    ]
    log.info(f"Building {label} image: {target}")
    subprocess.run(cmd, check=True, cwd=SCRIPT_DIR, capture_output=not log.VERBOSE)
    log.result(f"Built {target}")

    push_cmd = tdfs_cmd(cfg, SCRIPT_DIR) + ["image", "push", "--force-http", target]
    log.info(f"Pushing {target}")
    subprocess.run(push_cmd, check=True, cwd=SCRIPT_DIR, capture_output=not log.VERBOSE)
    log.result(f"Pushed {target}")


def build_and_push_2dfs(allotments: list[list[str]], source_image: str, cfg: EnvConfig, artifacts_dir: str | None = None) -> None:
    build_and_push_2dfs_image(
        allotments, cfg,
        target=build_name_2dfs(source_image, cfg),
        base_image=plain_base_image(source_image, cfg),
        extra_flags=[],
        label="2dfs",
        artifacts_dir=artifacts_dir,
    )


def build_and_push_2dfs_stargz(allotments: list[list[str]], source_image: str, cfg: EnvConfig, artifacts_dir: str | None = None) -> None:
    build_and_push_2dfs_image(
        allotments, cfg,
        target=build_name_2dfs_stargz(source_image, cfg),
        base_image=stargz_base_image(source_image, cfg),
        extra_flags=["--enable-stargz", "--stargz-chunk-size", "2097152"],  # 2 MiB (most optimal)
        label="2dfs-stargz",
        artifacts_dir=artifacts_dir,
    )


def build_and_push_2dfs_stargz_zstd(allotments: list[list[str]], source_image: str, cfg: EnvConfig, artifacts_dir: str | None = None) -> None:
    build_and_push_2dfs_image(
        allotments, cfg,
        target=build_name_2dfs_stargz_zstd(source_image, cfg),
        base_image=zstd_base_image(source_image, cfg),
        extra_flags=["--enable-stargz", "--use-zstd", "--stargz-chunk-size", "8388608"],  # 8 MiB (most optimal)
        label="2dfs-stargz-zstd",
        artifacts_dir=artifacts_dir,
    )


def build_and_push_stargz(allotments: list[list[str]], source_image: str, cfg: EnvConfig, artifacts_dir: str | None = None) -> None:
    create_stargz_dockerfile(allotments, stargz_base_image(source_image, cfg), SCRIPT_DIR)
    if artifacts_dir:
        snapshot_artifacts(SCRIPT_DIR, artifacts_dir)
    target = build_name_stargz(source_image, cfg)

    # force-compression=true makes sure the split layers are converted to stargz
    # otherwise cached layers are used which might not be compressed
    cmd = [
        "sudo", "buildctl", "build",
        "--frontend", "dockerfile.v0",
        "--opt", "filename=Dockerfile.stargz",
        "--local", f"context={SCRIPT_DIR}",
        "--local", f"dockerfile={SCRIPT_DIR}",
        "--output", f"type=image,name={target},push=true,compression=estargz,compression-level=1,force-compression=true,oci-mediatypes=true,registry.insecure=true",
    ]
    log.info(f"Building and pushing stargz image: {target}")
    subprocess.run(cmd, check=True, cwd=SCRIPT_DIR, capture_output=not log.VERBOSE)
    log.result(f"Built and pushed {target}")


def build_and_push_base(allotments: list[list[str]], base_splits: list[int], source_image: str, cfg: EnvConfig, artifacts_dir_fn=None) -> None:
    for r in base_splits:
        create_base_dockerfile(allotments[:r], plain_base_image(source_image, cfg), SCRIPT_DIR)
        if artifacts_dir_fn:
            snapshot_artifacts(SCRIPT_DIR, artifacts_dir_fn(r))
        target = build_name_base(source_image, cfg, r)

        cmd = [
            "sudo", "buildctl", "build",
            "--frontend", "dockerfile.v0",
            "--opt", "filename=Dockerfile.base",
            "--local", f"context={SCRIPT_DIR}",
            "--local", f"dockerfile={SCRIPT_DIR}",
            "--output", f"type=image,name={target},push=true,registry.insecure=true",
        ]
        log.info(f"Building and pushing base image: {target}")
        subprocess.run(cmd, check=True, cwd=SCRIPT_DIR, capture_output=not log.VERBOSE)
        log.result(f"Built and pushed {target}")


# ── per-mode public entry points (chunk_paths back-compat) ──────────


def prepare_2dfs(chunk_paths: list[str], source_image: str, cfg: EnvConfig, artifacts_dir: str | None = None) -> None:
    clear_2dfs_cache(cfg)
    build_and_push_2dfs([[p] for p in chunk_paths], source_image, cfg, artifacts_dir)


def prepare_2dfs_stargz(chunk_paths: list[str], source_image: str, cfg: EnvConfig, artifacts_dir: str | None = None) -> None:
    clear_2dfs_cache(cfg)
    build_and_push_2dfs_stargz([[p] for p in chunk_paths], source_image, cfg, artifacts_dir)


def prepare_2dfs_stargz_zstd(chunk_paths: list[str], source_image: str, cfg: EnvConfig, artifacts_dir: str | None = None) -> None:
    clear_2dfs_cache(cfg)
    build_and_push_2dfs_stargz_zstd([[p] for p in chunk_paths], source_image, cfg, artifacts_dir)


def prepare_stargz(chunk_paths: list[str], source_image: str, cfg: EnvConfig, artifacts_dir: str | None = None) -> None:
    build_and_push_stargz([[p] for p in chunk_paths], source_image, cfg, artifacts_dir)


def prepare_base(chunk_paths: list[str], base_splits: list[int], source_image: str, cfg: EnvConfig, artifacts_dir_fn=None) -> None:
    build_and_push_base([[p] for p in chunk_paths], base_splits, source_image, cfg, artifacts_dir_fn)
