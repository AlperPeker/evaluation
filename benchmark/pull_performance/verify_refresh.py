import hashlib
import os
import subprocess
import uuid
from datetime import datetime, timezone

from shared import log
from shared.artifacts import (
    chunks_to_groups,
    clear_artifacts,
    mutate_chunk,
    write_2dfs_json,
)
from shared.config import load_config
from shared.registry import (
    clear_registry,
    fetch_layer_digests,
    image_slug,
    prepare_local_registry,
    registry,
    save_toc,
    tdfs_cmd,
)
from shared.services import clear_2dfs_cache, clear_stargz_cache, ensure_buildkit
from pull_performance.measure import _next_container_name
from pull_performance.prepare import prepare_chunks
from pull_performance.refresh_common import (
    base_image,
    extra_flags,
    start_container,
    stop_container,
    timed_pull,
    cat_chunks_in_container,
)

# Verifies that `ctr-remote refresh` invalidates the FUSE/stargz cache for a layer
# whose blob digest changed, so a file in the running container returns the new
# content on next read without restarting the container.
# Flow: build v0 → rpull → warm cache (cat all chunks) → sha256 chunk1 (old) →
# mutate chunk1 → build v1 → ctr-remote refresh → sha256 chunk1 (must equal v1).

CFG = load_config()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODE = "2dfs-stargz"
MODEL = "openai-community/gpt2"
SOURCE_IMAGE = "docker.io/library/python:3.12-slim"
NUM_CHUNKS = 12
NUM_LAYERS = 3
MUTATED_IDX = 0


def _build_target() -> str:
    return f"{registry(CFG)}/{image_slug(SOURCE_IMAGE)}-{MODE}-verify-refresh:latest"


def _pull_ref() -> str:
    end_col = NUM_LAYERS - 1
    return f"{registry(CFG)}/library/{image_slug(SOURCE_IMAGE)}-{MODE}-verify-refresh:latest--0.0.0.{end_col}"


def _build_and_push(chunk_paths: list[str], label: str) -> None:
    target = _build_target()
    base = base_image(SOURCE_IMAGE, CFG, MODE)
    groups = chunks_to_groups(chunk_paths, NUM_LAYERS)
    write_2dfs_json(groups, SCRIPT_DIR)

    cmd = tdfs_cmd(CFG, SCRIPT_DIR) + [
        "build", "--platforms", "linux/amd64",
        *extra_flags(MODE),
        "--force-http", "-f", "2dfs.json",
        base, target,
    ]
    log.info(f"Building {label}: {target}")
    subprocess.run(cmd, check=True, cwd=SCRIPT_DIR, capture_output=not log.VERBOSE)
    log.result(f"Built {target}")

    push_cmd = tdfs_cmd(CFG, SCRIPT_DIR) + ["image", "push", "--force-http", target]
    log.info(f"Pushing {target}")
    subprocess.run(push_cmd, check=True, cwd=SCRIPT_DIR, capture_output=not log.VERBOSE)
    log.result(f"Pushed {target}")


def _sha256_in_container(name: str, path: str) -> str:
    exec_id = uuid.uuid4().hex[:8]
    result = subprocess.run(
        ["sudo", "ctr", "tasks", "exec", "--exec-id", exec_id, name,
         "sh", "-c", f"sha256sum {path}"],
        check=True, capture_output=True, text=True,
    )
    return result.stdout.split()[0]


def _sha256_local(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _repo() -> str:
    return f"library/{image_slug(SOURCE_IMAGE)}-{MODE}-verify-refresh"


def _full_tag() -> str:
    end_col = NUM_LAYERS - 1
    return f"latest--0.0.0.{end_col}"


def main():
    log.set_verbose(True)
    clear_artifacts(SCRIPT_DIR)
    ensure_buildkit()

    execution_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    toc_dir = os.path.join(SCRIPT_DIR, "artifacts", "verify-refresh", execution_ts)
    os.makedirs(toc_dir, exist_ok=True)
    log.result(f"TOC artifacts -> {toc_dir}")

    chunk_paths = prepare_chunks(MODEL, NUM_CHUNKS)
    prepare_local_registry(SOURCE_IMAGE, registry(CFG))

    clear_registry(CFG, preserve_base=True, verbose=False)
    clear_2dfs_cache(CFG)
    clear_stargz_cache()

    name = None
    mutated = False
    try:
        log.result("=== Build + push v0 ===")
        _build_and_push(chunk_paths, "v0")
        repo = _repo()
        v0_digests = fetch_layer_digests(registry(CFG), repo, _full_tag())
        log.info(f"v0 layers: {[d[:19] for d in v0_digests]}")
        for i, d in enumerate(v0_digests):
            save_toc(registry(CFG), repo, d, os.path.join(toc_dir, f"toc_v0_layer{i}.json"))

        pull_ref = _pull_ref()
        log.result(f"=== rpull v0: {pull_ref} ===")
        timed_pull(["sudo", "ctr-remote", "images", "rpull", "--plain-http", pull_ref])

        name = _next_container_name(f"verify-refresh-{MODE.replace('-', '')}")
        start_container(pull_ref, name)

        log.info(f"Cat all {NUM_CHUNKS} files in container...")
        cat_chunks_in_container(name, NUM_CHUNKS)

        target_file = f"/chunk{MUTATED_IDX + 1}.bin"
        pre_digest = _sha256_in_container(name, target_file)
        log.result(f"pre-refresh  in-container sha256({target_file}) = {pre_digest}")

        log.result(f"=== Mutate {os.path.basename(chunk_paths[MUTATED_IDX])} + build v1 ===")
        mutate_chunk(chunk_paths[MUTATED_IDX])
        mutated = True
        expected_digest = _sha256_local(chunk_paths[MUTATED_IDX])
        log.info(f"expected post-refresh sha256({target_file}) = {expected_digest}")
        _build_and_push(chunk_paths, "v1")
        v1_digests = fetch_layer_digests(registry(CFG), repo, _full_tag())
        log.info(f"v1 layers: {[d[:19] for d in v1_digests]}")

        changed = [
            i for i in range(min(len(v0_digests), len(v1_digests)))
            if v0_digests[i] != v1_digests[i]
        ]
        if not changed:
            log.result("WARN: no changed layer detected between v0 and v1")
        for i in changed:
            save_toc(registry(CFG), repo, v1_digests[i], os.path.join(toc_dir, f"toc_v1_layer{i}.json"))

        log.result(f"=== ctr-remote refresh {pull_ref} ===")
        refresh = subprocess.run(
            ["sudo", "ctr-remote", "refresh", "--plain-http", pull_ref],
            capture_output=True, text=True,
        )
        if refresh.stdout:
            print(refresh.stdout, end="")
        if refresh.stderr:
            print(refresh.stderr, end="")
        if refresh.returncode != 0:
            log.result(f"FAIL: refresh exited with code {refresh.returncode}")
            return

        post_digest = _sha256_in_container(name, target_file)
        log.result(f"post-refresh in-container sha256({target_file}) = {post_digest}")

        if pre_digest == post_digest:
            log.result(f"FAIL: digest unchanged after refresh ({pre_digest})")
        elif post_digest != expected_digest:
            log.result(f"FAIL: post-refresh digest {post_digest} != expected {expected_digest}")
        else:
            log.result(f"PASS: {target_file} refreshed successfully ({pre_digest[:12]}... -> {post_digest[:12]}...)")
    finally:
        if name is not None:
            stop_container(name)
        if mutated:
            log.info("Restoring mutated chunk...")
            mutate_chunk(chunk_paths[MUTATED_IDX])
        clear_registry(CFG, preserve_base=True, verbose=False)
        clear_artifacts(SCRIPT_DIR)


if __name__ == "__main__":
    main()
