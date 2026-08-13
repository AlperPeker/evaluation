import hashlib
import os
import subprocess
import time
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
    image_slug,
    prepare_local_registry,
    registry,
    tdfs_cmd,
)
from shared.services import (
    clear_2dfs_cache,
    clear_stargz_cache,
    ensure_buildkit,
    save_stargz_run_log,
)
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

# Verifies that `ctr-remote watch` subscribes to an image tag and, when a new
# version is pushed, auto-refreshes the FUSE/stargz cache so a file in the
# running container returns the new content without an explicit refresh call.
# Flow: build v0 → rpull → warm cache → sha256 chunk1 (old) → ctr-remote watch
# → wait → mutate chunk1 → build/push v1 → wait → sha256 chunk1 (must equal v1).

CFG = load_config()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODE = "2dfs-stargz"
MODEL = "openai-community/gpt2"
SOURCE_IMAGE = "docker.io/library/python:3.12-slim"
NUM_CHUNKS = 12
NUM_LAYERS = 3
MUTATED_IDX = 0
WATCH_WAIT_S = 30


def _build_target() -> str:
    return f"{registry(CFG)}/{image_slug(SOURCE_IMAGE)}-{MODE}-verify-watch:latest"


def _pull_ref() -> str:
    end_col = NUM_LAYERS - 1
    return f"{registry(CFG)}/library/{image_slug(SOURCE_IMAGE)}-{MODE}-verify-watch:latest--0.0.0.{end_col}"


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


def _watch(pull_ref: str) -> None:
    log.info(f"watch --with-background-fetch {pull_ref}")
    result = subprocess.run(
        ["sudo", "ctr-remote", "watch", "--with-background-fetch", pull_ref],
        capture_output=True, text=True,
    )
    log.result(f"--- watch output (rc={result.returncode}) ---")
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")
    log.result("--- end watch output ---")
    if result.returncode != 0:
        raise RuntimeError(f"watch failed (rc={result.returncode})")


def _watch_list() -> None:
    result = subprocess.run(
        ["sudo", "ctr-remote", "watch-list"],
        capture_output=True, text=True,
    )
    log.result(f"--- watch-list output (rc={result.returncode}) ---")
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")
    log.result("--- end watch-list output ---")


def _unwatch(pull_ref: str) -> None:
    log.info(f"unwatch {pull_ref}")
    result = subprocess.run(
        ["sudo", "ctr-remote", "unwatch", pull_ref],
        capture_output=True, text=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")
    if result.returncode != 0:
        log.result(f"WARN: unwatch exited with code {result.returncode}")


def main():
    log.set_verbose(True)
    clear_artifacts(SCRIPT_DIR)
    ensure_buildkit()

    chunk_paths = prepare_chunks(MODEL, NUM_CHUNKS)
    prepare_local_registry(SOURCE_IMAGE, registry(CFG))

    clear_registry(CFG, preserve_base=True, verbose=False)
    clear_2dfs_cache(CFG)
    clear_stargz_cache()

    log.result("=== Build + push v0 ===")
    _build_and_push(chunk_paths, "v0")

    pull_ref = _pull_ref()
    log.result(f"=== rpull v0: {pull_ref} ===")
    timed_pull(["sudo", "ctr-remote", "images", "rpull", "--plain-http", pull_ref])

    name = _next_container_name(f"verify-watch-{MODE.replace('-', '')}")
    start_container(pull_ref, name)

    log.info(f"Cat all {NUM_CHUNKS} files in container...")
    cat_chunks_in_container(name, NUM_CHUNKS)

    target_file = f"/chunk{MUTATED_IDX + 1}.bin"
    pre_digest = _sha256_in_container(name, target_file)
    log.result(f"pre-watch  in-container sha256({target_file}) = {pre_digest}")

    log_window_start = time.time()
    log.result(f"=== Subscribe: ctr-remote watch {pull_ref} ===")
    try:
        _watch(pull_ref)
    except RuntimeError as e:
        log.result(f"FAIL: {e}")
        stop_container(name)
        return
    _watch_list()
    # watch needs 30s to register first "baseline" manifest digest
    time.sleep(WATCH_WAIT_S)
    _watch_list()
    log.result(f"=== Mutate {os.path.basename(chunk_paths[MUTATED_IDX])} + build v1 ===")
    mutate_chunk(chunk_paths[MUTATED_IDX])
    expected_digest = _sha256_local(chunk_paths[MUTATED_IDX])
    log.info(f"expected post-watch sha256({target_file}) = {expected_digest}")
    _build_and_push(chunk_paths, "v1")

    log.result(f"=== Sleep {WATCH_WAIT_S * 2}s for watcher to detect + bg-fetch ===")
    time.sleep(WATCH_WAIT_S * 2)

    post_digest = _sha256_in_container(name, target_file)
    log.result(f"post-watch in-container sha256({target_file}) = {post_digest}")

    if pre_digest == post_digest:
        log.result(f"FAIL: digest unchanged after watch ({pre_digest})")
    elif post_digest != expected_digest:
        log.result(f"FAIL: post-watch digest {post_digest} != expected {expected_digest}")
    else:
        log.result(f"PASS: {target_file} updated by watcher ({pre_digest[:12]}... -> {post_digest[:12]}...)")

    _watch_list()
    _unwatch(pull_ref)
    stop_container(name)

    run_end_s = time.time()
    execution_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    log_path = os.path.join(SCRIPT_DIR, "logs", "verify-watch", execution_ts, "stargz.json")
    save_stargz_run_log(log_window_start, run_end_s, log_path)
    log.result(f"stargz logs -> {log_path}")

    log.info("Restoring mutated chunk...")
    mutate_chunk(chunk_paths[MUTATED_IDX])

    clear_registry(CFG, preserve_base=True, verbose=False)
    clear_artifacts(SCRIPT_DIR)


if __name__ == "__main__":
    main()
