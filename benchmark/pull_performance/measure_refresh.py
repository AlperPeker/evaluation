import os
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from typing import Callable, Literal

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

from shared import cv_splits as cv
from shared import log, paths
from shared.artifacts import (
    clear_artifacts, snapshot_artifacts, write_2dfs_json, xor_first_byte,
)
from shared.charts import figure_footer, save_figure, write_csv
from shared.config import load_config
from shared.prometheus import bytes_by_layer
from shared.registry import (
    clear_registry, fetch_layer_digests, fetch_layer_sizes, image_slug,
    prepare_local_registry, registry, save_toc, tdfs_cmd,
)
from shared.services import (
    clear_2dfs_cache, clear_overlayfs_cache, clear_stargz_cache, ensure_buildkit,
    prune_buildkit, save_stargz_run_log,
)
from shared.stargz_config import read_base_config
from pull_performance.measure import _next_container_name
from pull_performance.paths import (
    refresh_artifacts_dir, refresh_bytes_chart_path, refresh_bytes_csv_path,
    refresh_chart_path, refresh_csv_path, refresh_log_path,
    refresh_merged_csv_path, refresh_merged_bytes_csv_path,
    refresh_multimodel_merged_csv_path, refresh_multimodel_merged_bytes_csv_path,
    refresh_stargz_config_path, refresh_run_metadata_path,
)
from shared.run_metadata import write_run_json
from pull_performance.refresh_common import (
    base_image, build_mode, extra_flags, start_container, stop_container,
    timed_pull,
)

load_dotenv()

CFG = load_config()
VERBOSE = False
N_RUNS = CFG.refresh_n_runs
LAZY_MODE = "2dfs-stargz"   # used by manual-lazy + refresh
NO_LAZY_MODE = "oci"        # used by manual-oci (full pull)

# Single-model config (chat_template mutation on a tokenizer JSON).
MUTATED_FILENAME = "tokenizer_config.json"
MUTATED_FIELD = "chat_template"
MUTATION_STRING = b"added string"
WEIGHT_SUFFIXES = (".safetensors", ".bin")  # safetensors or PyTorch pickle weights

OP_TYPES = ["on_demand_bytes_fetched"]
PROM_SETTLE_S = 1.0
SCHEMA_VERSION = 4
MULTI_SCHEMA_VERSION = 1

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

UpdateStrategy = Literal["manual-lazy", "manual-oci", "refresh"]
ExperimentPhase = Literal["setup", "update"]
ImageVersion = Literal["before", "after"]
StrategyRunner = Callable[["RefreshExperiment", str | None], "StrategyResult"]


# ── experiment configs ──────────────────────────────────────────────────


@dataclass(frozen=True)
class SingleModel:
    """A single HuggingFace model packed into one allotment; a config JSON field
    is mutated to produce the after-image."""
    hf_model: str
    base_image: str


@dataclass(frozen=True)
class MultiModel:
    """Several pre-split CV models packed one-allotment-per-model into one image.
    One model (modified_model) is mutated across ALL its files to produce
    the after-image; the other allotments are byte-identical before vs after."""
    label: str
    split_dirs: list[str]   # stack order, bottom -> top
    modified_model: str     # must be one of split_dirs
    base_image: str


EXPERIMENTS = [
    # SingleModel("Qwen/Qwen3.5-9B", "docker.io/ollama/ollama"),
    MultiModel(
        "cv-4model",
        ["resnet50_seperated", "deeplab_v3_seperated",
         "efficientnet_v2M_seperated", "yolov3_seperated"],
        modified_model="resnet50_seperated",
        base_image="docker.io/tensorflow/tensorflow",
    ),
]


@dataclass(frozen=True)
class RefreshTimeRow:
    schema_version: int
    model: str
    base_image: str
    mode: str
    run: int
    update_strategy: UpdateStrategy
    setup_cold_read_s: float | None
    setup_warm_read_s: float | None
    stop_total_s: float | None
    stop_kill_s: float | None
    stop_task_delete_s: float | None
    stop_container_delete_s: float | None
    pull_s: float | None
    run_s: float | None
    refresh_s: float | None
    read_s: float
    total_s: float


@dataclass(frozen=True)
class RefreshTimeRowMultimodal:
    schema_version: int
    label: str
    base_image: str
    num_models: int
    models: str             # ordered stack names, e.g. "resnet50|deeplab_v3"
    modified_model: str
    mode: str
    run: int
    update_strategy: UpdateStrategy
    setup_cold_read_s: float | None
    setup_warm_read_s: float | None
    stop_total_s: float | None
    stop_kill_s: float | None
    stop_task_delete_s: float | None
    stop_container_delete_s: float | None
    pull_s: float | None
    run_s: float | None
    refresh_s: float | None
    read_s: float
    total_s: float


@dataclass(frozen=True)
class RefreshBytesRow:
    schema_version: int
    model: str
    base_image: str
    mode: str
    run: int
    update_strategy: UpdateStrategy
    experiment_phase: ExperimentPhase
    layer: str
    op_type: str
    bytes: int


@dataclass(frozen=True)
class RefreshBytesRowMultimodal:
    schema_version: int
    label: str
    base_image: str
    num_models: int
    models: str
    modified_model: str
    mode: str
    run: int
    update_strategy: UpdateStrategy
    experiment_phase: ExperimentPhase
    layer: str
    op_type: str
    bytes: int


@dataclass(frozen=True)
class SetupResult:
    """Outcome of the before-image warm-up phase (shared by both strategies)."""
    name: str
    cold_read_s: float
    warm_read_s: float
    bytes_fetched: dict[str, dict[str, int]]


@dataclass(frozen=True)
class StrategyResult:
    """Outcome of one update strategy. Phase timings are populated per-strategy:
    manual-lazy sets stop/pull/run; manual-oci sets pull/run (no before
    container to stop); refresh sets refresh_s. setup is None for manual-oci."""
    setup: SetupResult | None
    update_bytes_fetched: dict[str, dict[str, int]]
    read_s: float
    total_s: float
    stop_total_s: float | None = None
    stop_kill_s: float | None = None
    stop_task_delete_s: float | None = None
    stop_container_delete_s: float | None = None
    pull_s: float | None = None
    run_s: float | None = None
    refresh_s: float | None = None


@dataclass(frozen=True)
class RefreshExperiment:
    """A resolved experiment the build/measure/output code runs, uniform across arms."""
    label: str
    base_image: str
    name_slug: str                  # registry image-name slug
    is_multi: bool
    groups: list[list[str]]         # allotments for write_2dfs_json
    in_paths: list[str]             # every file to cat
    size_str: str | None
    mutate: Callable[[], Callable[[], None]]  # apply after-image change, return its undo
    assert_mutated: Callable[[str, bool], None]  # (container, expect_mutated)
    meta: dict
    make_time_row: Callable[[int, UpdateStrategy, str, StrategyResult], object]
    make_bytes_rows: Callable[
        [int, UpdateStrategy, str, ExperimentPhase, dict[str, dict[str, int]]], list
    ]
    probe_disk_byte: Callable[[], int | None] | None = None  # rep file first byte on disk


# ── single-model snapshot download + mutation ───────────────────────────


def _model_snapshot_dir(hf_model: str) -> str:
    return paths.models_dir(SCRIPT_DIR, hf_model)


def download_snapshot(hf_model: str) -> list[str]:
    """Download full HF snapshot (weights + tokenizer/config JSONs).
    Returns absolute paths of every file in the snapshot dir.
    """
    local_dir = _model_snapshot_dir(hf_model)
    os.makedirs(local_dir, exist_ok=True)

    has_cfg = os.path.exists(os.path.join(local_dir, MUTATED_FILENAME))
    has_weights = any(
        f.endswith(WEIGHT_SUFFIXES) for f in os.listdir(local_dir)
    )
    if has_cfg and has_weights:
        log.info(f"Model snapshot present at {local_dir}, skipping download")
    else:
        log.info(f"Downloading full snapshot {hf_model} -> {local_dir}")
        token = os.environ.get("HF_TOKEN")
        snapshot_download(
            repo_id=hf_model,
            local_dir=local_dir,
            token=token,
            allow_patterns=["*.safetensors", "*.bin", "*.json", "*.txt", "*.model"],
        )

    files = sorted(
        os.path.join(local_dir, f)
        for f in os.listdir(local_dir)
        if os.path.isfile(os.path.join(local_dir, f))
    )
    if not any(f.endswith(WEIGHT_SUFFIXES) for f in files):
        raise RuntimeError(
            f"no weight files {WEIGHT_SUFFIXES} found in {local_dir}; refresh "
            f"experiment requires large weight files to be meaningful"
        )
    return files


def _mutate_chat_template(hf_model: str) -> tuple[int, int]:
    """Insert MUTATION_STRING at the start of tokenizer_config.json's
    chat_template value. Returns (offset, length) so we can restore.
    """
    path = os.path.join(_model_snapshot_dir(hf_model), MUTATED_FILENAME)
    with open(path, "rb") as f:
        data = bytearray(f.read())

    marker = f'"{MUTATED_FIELD}":'.encode()
    i = data.find(marker)
    if i < 0:
        raise RuntimeError(f"{MUTATED_FIELD} not found in {path}")

    # Find opening quote of the chat_template string value.
    q = data.find(b'"', i + len(marker))
    if q < 0:
        raise RuntimeError(f"{MUTATED_FIELD} value quote not found in {path}")

    insert_at = q + 1
    data[insert_at:insert_at] = MUTATION_STRING
    with open(path, "wb") as f:
        f.write(data)
    log.info(
        f"Inserted {len(MUTATION_STRING)} bytes at offset {insert_at} in "
        f"{MUTATED_FILENAME}"
    )
    return insert_at, len(MUTATION_STRING)


def _restore_byte(hf_model: str, offset: int, length: int) -> None:
    path = os.path.join(_model_snapshot_dir(hf_model), MUTATED_FILENAME)
    with open(path, "rb") as f:
        data = bytearray(f.read())
    del data[offset:offset + length]
    with open(path, "wb") as f:
        f.write(data)
    log.info(f"Removed {length} bytes at offset {offset} in {MUTATED_FILENAME}")


def _assert_chat_template_mutated(name: str, expected: bool) -> None:
    """Verify whether MUTATION_STRING is present in MUTATED_FILENAME inside the
    container. Runs outside any timed window."""
    r = subprocess.run(
        ["sudo", "ctr", "tasks", "exec", "--exec-id", uuid.uuid4().hex[:8],
         name, "cat", f"/{MUTATED_FILENAME}"],
        capture_output=True, check=True,
    )
    found = MUTATION_STRING in r.stdout
    if found != expected:
        raise RuntimeError(
            f"validation failed in {name}: expected mutated={expected}, got {found}"
        )
    log.result(f"  validation OK ({name}): mutated={found}")


# ── multi-model mutation ────────────────────────────────────────────────


def _mutate_model_files(model_files: list[str]) -> None:
    """XOR the first byte of every file of the modified model (reversible)."""
    for p in model_files:
        xor_first_byte(p)
    log.info(f"Mutated first byte of {len(model_files)} files of modified model")


def _assert_first_byte_mutated(
    name: str, in_path: str, orig_byte: int, expected: bool,
) -> None:
    """Verify the first byte of a representative modified file inside the
    container matches the expected image version. Runs outside any timed window."""
    r = subprocess.run(
        ["sudo", "ctr", "tasks", "exec", "--exec-id", uuid.uuid4().hex[:8],
         name, "head", "-c", "1", in_path],
        capture_output=True, check=True,
    )
    got = r.stdout[0] if r.stdout else None
    want = (orig_byte ^ 0xFF) if expected else orig_byte
    if got != want:
        raise RuntimeError(
            f"validation failed in {name}: {in_path} first byte expected "
            f"{want} (mutated={expected}), got {got}"
        )
    log.result(f"  validation OK ({name}): mutated={expected}")


# ── image naming ────────────────────────────────────────────────────────


def _repo(exp: RefreshExperiment, mode: str = LAZY_MODE) -> str:
    return f"library/{exp.name_slug}-{mode}-refresh"


def _build_target(exp: RefreshExperiment, image_version: ImageVersion, mode: str = LAZY_MODE) -> str:
    return f"{registry(CFG)}/{exp.name_slug}-{mode}-refresh:{image_version}"


def _partition_tag(exp: RefreshExperiment) -> str:
    """Full-content partition (all cols) — needed by OCI so no columns are
    dropped; a no-op for stargz under noprefetch=true."""
    return f"--0.0.0.{len(exp.groups) - 1}"


def _pull_ref(exp: RefreshExperiment, image_version: ImageVersion, mode: str = LAZY_MODE) -> str:
    return f"{registry(CFG)}/library/{exp.name_slug}-{mode}-refresh:{image_version}{_partition_tag(exp)}"


# ── TOC export ───────────────────────────────────────────────────────────


def _export_tocs(exp: RefreshExperiment, image_version: ImageVersion, toc_dir: str) -> list[str]:
    os.makedirs(toc_dir, exist_ok=True)
    repo = _repo(exp)
    digests = fetch_layer_digests(registry(CFG), repo, f"{image_version}{_partition_tag(exp)}")
    log.info(f"{image_version} layers: {[d[:19] for d in digests]}")
    for i, d in enumerate(digests):
        save_toc(
            registry(CFG), repo, d,
            os.path.join(toc_dir, f"toc_{image_version}_layer{i}.json"),
        )
    return digests


# ── build helpers ────────────────────────────────────────────────────────


def _build_version(
    exp: RefreshExperiment, image_version: ImageVersion, mode: str = LAZY_MODE,
) -> None:
    target = _build_target(exp, image_version, mode)
    base = base_image(exp.base_image, CFG, mode)

    write_2dfs_json(exp.groups, SCRIPT_DIR)

    cmd = tdfs_cmd(CFG, SCRIPT_DIR) + [
        "build", "--platforms", "linux/amd64",
        *extra_flags(mode),
        "--force-http", "-f", "2dfs.json",
        base, target,
    ]
    log.info(f"Building {image_version}: {target}")
    subprocess.run(cmd, check=True, cwd=SCRIPT_DIR, capture_output=not log.VERBOSE)
    log.result(f"Built {target}")

    push_cmd = tdfs_cmd(CFG, SCRIPT_DIR) + ["image", "push", "--force-http", target]
    log.info(f"Pushing {target}")
    subprocess.run(push_cmd, check=True, cwd=SCRIPT_DIR, capture_output=not log.VERBOSE)
    log.result(f"Pushed {target}")


def _build_mode_pair(
    exp: RefreshExperiment,
    mode: str,
    artifacts_dir: str | None = None,
    toc_dir: str | None = None,
) -> None:
    """Build and push before + after images for one mode (mutates for after,
    restores it in a finally block)."""
    def _probe(when: str) -> None:
        if exp.probe_disk_byte:
            log.result(f"PROBE [{when}] rep first byte on disk = {exp.probe_disk_byte()}")

    _probe("pre-before-build")
    _build_version(exp, "before", mode)
    if artifacts_dir:
        snapshot_artifacts(SCRIPT_DIR, artifacts_dir)
    before_digests: list[str] = []
    if toc_dir:
        before_digests = _export_tocs(exp, "before", toc_dir)

    _probe("post-before-build")
    undo = exp.mutate()
    _probe("post-mutate")
    try:
        _probe("pre-after-build")
        _build_version(exp, "after", mode)
    finally:
        undo()

    if toc_dir:
        after_digests = _export_tocs(exp, "after", toc_dir)
        changed = [
            i for i in range(min(len(before_digests), len(after_digests)))
            if before_digests[i] != after_digests[i]
        ]
        log.result(
            f"TOC: before vs after changed layer indices: {changed}"
            if changed else "TOC: no changed layers detected between before and after"
        )


# ── measurement helpers ──────────────────────────────────────────────────


def _container_paths(files: list[str]) -> list[str]:
    """In-container paths: files land at /{basename} per write_2dfs_json layout."""
    return [f"/{os.path.basename(p)}" for p in files]


def _cat_all_in_container(name: str, in_paths: list[str]) -> float:
    """Exec cat over all files inside container, return elapsed seconds."""
    exec_id = uuid.uuid4().hex[:8]
    files = " ".join(in_paths)
    log.info(f"Reading {len(in_paths)} files in {name}")
    start = time.perf_counter()
    subprocess.run(
        ["sudo", "ctr", "tasks", "exec", "--exec-id", exec_id,
         name, "sh", "-c", f"cat {files} > /dev/null"],
        check=True, capture_output=not log.VERBOSE,
    )
    return time.perf_counter() - start


def _snapshot_bytes() -> dict[str, dict[str, int]]:
    """Returns {op_type: {layer: bytes}} at current time."""
    return {op: bytes_by_layer(op) for op in OP_TYPES}


def _bytes_fetched_delta(before: dict[str, dict[str, int]],
                         after: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    """Per-op_type, per-layer bytes fetched between two counter snapshots.
    Layers present only in `after` count as new."""
    out: dict[str, dict[str, int]] = {}
    for op in OP_TYPES:
        b = before.get(op, {})
        a = after.get(op, {})
        d: dict[str, int] = {}
        for layer in set(b) | set(a):
            v = a.get(layer, 0) - b.get(layer, 0)
            if v != 0:
                d[layer] = v
        out[op] = d
    return out


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def _model_size_str(hf_model: str) -> str | None:
    d = _model_snapshot_dir(hf_model)
    if not os.path.isdir(d):
        return None
    total = 0
    for f in os.listdir(d):
        p = os.path.join(d, f)
        if os.path.isfile(p):
            total += os.path.getsize(p)
    return _fmt_bytes(total) if total > 0 else None


def _groups_size_str(groups: list[list[str]]) -> str | None:
    total = sum(os.path.getsize(p) for g in groups for p in g)
    return _fmt_bytes(total) if total > 0 else None


def _snapshot_stats(snapshot_files: list[str]) -> dict:
    """Model snapshot summary for run metadata (sizes in MB)."""
    sizes = [os.path.getsize(p) for p in snapshot_files]
    weights = [s for p, s in zip(snapshot_files, sizes) if p.endswith(WEIGHT_SUFFIXES)]
    return {
        "num_files": len(snapshot_files),
        "num_weights": len(weights),
        "total_mb": round(sum(sizes) / (1024 ** 2), 1),
        "weights_mb": round(sum(weights) / (1024 ** 2), 1),
    }


def _log_bytes_fetched(label: str, bytes_fetched: dict[str, dict[str, int]]) -> None:
    for op in OP_TYPES:
        total = sum(bytes_fetched.get(op, {}).values())
        log.result(f"  {label} {op}: total={_fmt_bytes(total)} "
                   f"layers={len(bytes_fetched.get(op, {}))}")


def _setup(exp: RefreshExperiment, mode: str = LAZY_MODE) -> SetupResult:
    """Clear stargz cache, pull before-image, start container, warm page cache
    via cat.

    LAZY_MODE pulls lazily (rpull, stargz snapshotter); NO_LAZY_MODE pulls the
    full image (overlayfs). Setup bytes are the on-demand bytes fetched during
    warm-up — always 0 for NO_LAZY_MODE since the full image is already local.

    Diagnostic: cat twice back-to-back. If second cat is fast, the kernel page
    cache is being used; if both are ~same, FUSE is bypassing it (in which case
    refresh cannot help regardless of mount preservation).
    """
    lazy = mode == LAZY_MODE
    snapshotter = "stargz" if lazy else "overlayfs"
    clear_stargz_cache()
    if not lazy:
        clear_overlayfs_cache()
    time.sleep(PROM_SETTLE_S)
    bytes_before = _snapshot_bytes()
    before_ref = _pull_ref(exp, "before", mode)
    log.info(f"Pulling before-image (setup, {mode}): {before_ref}")
    if lazy:
        pull_cmd = ["sudo", "ctr-remote", "images", "rpull",
                    "--plain-http", before_ref]
    else:
        pull_cmd = ["sudo", "ctr-remote", "images", "pull",
                    "--snapshotter", "overlayfs", "--plain-http", before_ref]
    subprocess.run(pull_cmd, check=True, capture_output=not log.VERBOSE)
    name = _next_container_name("refresh-before")
    start_container(before_ref, name, snapshotter=snapshotter)

    cold_t = _cat_all_in_container(name, exp.in_paths)
    warm_t = _cat_all_in_container(name, exp.in_paths)
    exp.assert_mutated(name, False)
    time.sleep(PROM_SETTLE_S)
    setup_bytes_fetched = _bytes_fetched_delta(bytes_before, _snapshot_bytes())
    log.result(
        f"DIAGNOSTICS: warm-up cat#1 (cold)={cold_t:.2f}s "
        f"cat#2 (re-read)={warm_t:.2f}s"
    )
    if cold_t > 0:
        ratio = warm_t / cold_t
        log.result(
            f"DIAGNOSTICS: re-read/cold ratio={ratio:.2f} "
            f"({'kernel page cache effective' if ratio < 0.5 else 'kernel page cache NOT effective — FUSE likely bypasses it'})"
        )
    _log_bytes_fetched("setup", setup_bytes_fetched)
    return SetupResult(name, cold_t, warm_t, bytes_fetched=setup_bytes_fetched)


def _oci_after_layer_bytes(exp: RefreshExperiment) -> dict[str, int]:
    """Per-layer compressed sizes of the whole OCI after-image, from the
    registry manifest. Represents the full image a no-lazy pull downloads."""
    return fetch_layer_sizes(
        registry(CFG), _repo(exp, NO_LAZY_MODE),
        f"after{_partition_tag(exp)}",
    )


def _run_manual_oci_strategy(
    exp: RefreshExperiment, log_path: str | None = None,
) -> StrategyResult:
    """No lazy loading: stop -> full pull after-image -> run -> read.
    Mirrors manual-lazy but pulls the whole image over overlayfs."""
    log_window_start = time.time()
    setup = _setup(exp, NO_LAZY_MODE)
    name = setup.name
    after_ref = _pull_ref(exp, "after", NO_LAZY_MODE)

    # overlayfs rootfs unmounts cheaply, so this stop should be fast (vs stargz FUSE).
    kill_t, task_del_t, container_del_t = stop_container(name)
    stop_t = kill_t + task_del_t + container_del_t

    log.info(f"Full-pulling OCI after-image: {after_ref}")
    pull_t = timed_pull(
        ["sudo", "ctr-remote", "images", "pull",
         "--snapshotter", "overlayfs", "--plain-http", after_ref]
    )

    name2 = _next_container_name("refresh-oci-after")
    t0 = time.perf_counter()
    start_container(after_ref, name2, snapshotter="overlayfs")
    run_t = time.perf_counter() - t0

    read_t = _cat_all_in_container(name2, exp.in_paths)
    exp.assert_mutated(name2, True)
    update_bytes_fetched = {OP_TYPES[0]: _oci_after_layer_bytes(exp)}
    stop_container(name2)
    log.result(
        f"  manual-oci: stop={stop_t:.2f}s (kill={kill_t:.2f} task-del={task_del_t:.2f} "
        f"container-del={container_del_t:.2f}) pull={pull_t:.2f}s "
        f"run={run_t:.2f}s read={read_t:.2f}s"
    )
    _log_bytes_fetched("manual-oci update", update_bytes_fetched)
    if log_path:
        save_stargz_run_log(log_window_start, time.time(), log_path)
        log.result(f"  stargz logs -> {log_path}")
    return StrategyResult(
        setup=setup,
        update_bytes_fetched=update_bytes_fetched,
        read_s=read_t,
        total_s=stop_t + pull_t + run_t + read_t,
        stop_total_s=stop_t,
        stop_kill_s=kill_t,
        stop_task_delete_s=task_del_t,
        stop_container_delete_s=container_del_t,
        pull_s=pull_t,
        run_s=run_t,
    )


def _run_manual_lazy_strategy(
    exp: RefreshExperiment, log_path: str | None = None,
) -> StrategyResult:
    """stop -> rpull after-image -> run -> read."""
    log_window_start = time.time()
    setup = _setup(exp)
    name = setup.name
    after_ref = _pull_ref(exp, "after")

    update_before = _snapshot_bytes()

    # Stop the old container before pulling the new image. The task delete
    # dominates: unmounting the FUSE rootfs blocks until the snapshotter tears
    # the mount down (~6s).
    kill_t, task_del_t, container_del_t = stop_container(name)
    stop_t = kill_t + task_del_t + container_del_t

    pull_t = timed_pull(
        ["sudo", "ctr-remote", "images", "rpull", "--plain-http", after_ref]
    )

    name2 = _next_container_name("refresh-after")
    t0 = time.perf_counter()
    start_container(after_ref, name2)
    run_t = time.perf_counter() - t0

    read_t = _cat_all_in_container(name2, exp.in_paths)
    exp.assert_mutated(name2, True)
    time.sleep(PROM_SETTLE_S)
    update_bytes_fetched = _bytes_fetched_delta(update_before, _snapshot_bytes())
    stop_container(name2)
    log.result(
        f"  manual-lazy: stop={stop_t:.2f}s (kill={kill_t:.2f} task-del={task_del_t:.2f} "
        f"container-del={container_del_t:.2f}) pull={pull_t:.2f}s "
        f"run={run_t:.2f}s read={read_t:.2f}s"
    )
    _log_bytes_fetched("manual-lazy update", update_bytes_fetched)
    if log_path:
        save_stargz_run_log(log_window_start, time.time(), log_path)
        log.result(f"  stargz logs -> {log_path}")
    return StrategyResult(
        setup=setup,
        update_bytes_fetched=update_bytes_fetched,
        read_s=read_t,
        total_s=stop_t + pull_t + run_t + read_t,
        stop_total_s=stop_t,
        stop_kill_s=kill_t,
        stop_task_delete_s=task_del_t,
        stop_container_delete_s=container_del_t,
        pull_s=pull_t,
        run_s=run_t,
    )


def _run_refresh_strategy(
    exp: RefreshExperiment, log_path: str | None = None,
) -> StrategyResult:
    """ctr-remote refresh before-image after-image -> read."""
    log_window_start = time.time()
    setup = _setup(exp)
    name = setup.name
    before_ref = _pull_ref(exp, "before")
    after_ref = _pull_ref(exp, "after")

    update_before = _snapshot_bytes()

    t0 = time.perf_counter()
    subprocess.run(
        ["sudo", "ctr-remote", "refresh", "--plain-http", before_ref, after_ref],
        check=True, capture_output=not log.VERBOSE,
    )
    refresh_t = time.perf_counter() - t0

    read_t = _cat_all_in_container(name, exp.in_paths)
    exp.assert_mutated(name, True)
    time.sleep(PROM_SETTLE_S)
    update_bytes_fetched = _bytes_fetched_delta(update_before, _snapshot_bytes())
    stop_container(name)
    log.result(f"  refresh:  refresh={refresh_t:.2f}s read={read_t:.2f}s")
    _log_bytes_fetched("refresh update", update_bytes_fetched)
    if log_path:
        save_stargz_run_log(log_window_start, time.time(), log_path)
        log.result(f"  stargz logs -> {log_path}")
    return StrategyResult(
        setup=setup,
        update_bytes_fetched=update_bytes_fetched,
        read_s=read_t,
        total_s=refresh_t + read_t,
        refresh_s=refresh_t,
    )


def measure_refresh(
    exp: RefreshExperiment, execution_ts: str,
    strategy_names: list[str] | None = None,
) -> tuple[list, list]:
    time_rows: list = []
    bytes_rows: list = []

    def record(run: int, strategy: UpdateStrategy, mode: str, r: StrategyResult) -> None:
        time_rows.append(exp.make_time_row(run, strategy, mode, r))
        if r.setup is not None:
            bytes_rows.extend(
                exp.make_bytes_rows(run, strategy, mode, "setup", r.setup.bytes_fetched)
            )
        bytes_rows.extend(
            exp.make_bytes_rows(run, strategy, mode, "update", r.update_bytes_fetched)
        )

    all_strategies: list[tuple[UpdateStrategy, str, StrategyRunner]] = [
        ("manual-lazy", LAZY_MODE, _run_manual_lazy_strategy),
        ("manual-oci", NO_LAZY_MODE, _run_manual_oci_strategy),
        ("refresh", LAZY_MODE, _run_refresh_strategy),
    ]
    active = [s for s in all_strategies if strategy_names is None or s[0] in strategy_names]

    for run in range(N_RUNS):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        log.info(f"\n[{ts}] === run {run + 1}/{N_RUNS} ===")

        rot = run % len(active)
        for strategy, mode, runner in active[rot:] + active[:rot]:
            log_path = refresh_log_path(
                SCRIPT_DIR, exp.label, exp.base_image, strategy, run, execution_ts,
            )
            record(run, strategy, mode, runner(exp, log_path))
            time.sleep(CFG.pull_cooldown)

    return time_rows, bytes_rows


# ── prepare ───────────────────────────────────────────────────────────────


def prepare_single(exp: SingleModel) -> RefreshExperiment:
    hf_model, source_image = exp.hf_model, exp.base_image
    snapshot_files = download_snapshot(hf_model)
    groups = [snapshot_files]
    in_paths = _container_paths(snapshot_files)
    meta = {
        "model": hf_model,
        "base_image": source_image,
        "mode": LAZY_MODE,
        "splits": _snapshot_stats(snapshot_files),
        "mutation": {
            "filename": MUTATED_FILENAME,
            "target": MUTATED_FIELD,
            "inserted_string": MUTATION_STRING.decode("utf-8", "replace"),
        },
    }

    def make_time_row(run, strategy, mode, r: StrategyResult) -> RefreshTimeRow:
        return RefreshTimeRow(
            schema_version=SCHEMA_VERSION, model=hf_model, base_image=source_image,
            mode=mode, run=run, update_strategy=strategy,
            setup_cold_read_s=r.setup.cold_read_s if r.setup else None,
            setup_warm_read_s=r.setup.warm_read_s if r.setup else None,
            stop_total_s=r.stop_total_s, stop_kill_s=r.stop_kill_s,
            stop_task_delete_s=r.stop_task_delete_s,
            stop_container_delete_s=r.stop_container_delete_s,
            pull_s=r.pull_s, run_s=r.run_s, refresh_s=r.refresh_s,
            read_s=r.read_s, total_s=r.total_s,
        )

    def make_bytes_rows(run, strategy, mode, phase, bytes_fetched) -> list[RefreshBytesRow]:
        rows: list[RefreshBytesRow] = []
        for op in OP_TYPES:
            for layer, b in sorted(bytes_fetched.get(op, {}).items()):
                rows.append(RefreshBytesRow(
                    schema_version=SCHEMA_VERSION, model=hf_model, base_image=source_image,
                    mode=mode, run=run, update_strategy=strategy,
                    experiment_phase=phase, layer=layer, op_type=op, bytes=b,
                ))
        return rows

    def mutate() -> Callable[[], None]:
        offset, length = _mutate_chat_template(hf_model)
        return lambda: _restore_byte(hf_model, offset, length)

    return RefreshExperiment(
        label=hf_model, base_image=source_image, name_slug=image_slug(source_image),
        is_multi=False, groups=groups, in_paths=in_paths,
        size_str=_model_size_str(hf_model), mutate=mutate,
        assert_mutated=_assert_chat_template_mutated,
        meta=meta, make_time_row=make_time_row, make_bytes_rows=make_bytes_rows,
    )


def prepare_multi(exp: MultiModel) -> RefreshExperiment:
    if exp.modified_model not in exp.split_dirs:
        raise ValueError(
            f"modified_model {exp.modified_model!r} not in split_dirs {exp.split_dirs}"
        )
    models = cv.prepare_cv_splits(exp.label, exp.split_dirs, SCRIPT_DIR)
    groups, _ = cv.pack_cv(models, [1] * len(models))  # one allotment per model
    names = [m.name for m in models]
    cv.print_cv_packing_table(exp.label, models, [100])

    modified_index = exp.split_dirs.index(exp.modified_model)
    modified_files = groups[modified_index]
    in_paths = _container_paths([p for g in groups for p in g])

    rep = modified_files[0]
    with open(rep, "rb") as f:
        orig_byte = f.read(1)[0]
    rep_in_path = f"/{os.path.basename(rep)}"

    meta = {
        "label": exp.label,
        "base_image": exp.base_image,
        "mode": LAZY_MODE,
        "models": names,
        "modified_model": exp.modified_model,
        "splits": cv.cv_split_stats(models),
        "mutation": {
            "modified_model": exp.modified_model,
            "num_files": len(modified_files),
            "method": "xor first byte of every file (reversible)",
        },
    }

    def make_time_row(run, strategy, mode, r: StrategyResult) -> RefreshTimeRowMultimodal:
        return RefreshTimeRowMultimodal(
            schema_version=MULTI_SCHEMA_VERSION, label=exp.label, base_image=exp.base_image,
            num_models=len(names), models="|".join(names), modified_model=exp.modified_model,
            mode=mode, run=run, update_strategy=strategy,
            setup_cold_read_s=r.setup.cold_read_s if r.setup else None,
            setup_warm_read_s=r.setup.warm_read_s if r.setup else None,
            stop_total_s=r.stop_total_s, stop_kill_s=r.stop_kill_s,
            stop_task_delete_s=r.stop_task_delete_s,
            stop_container_delete_s=r.stop_container_delete_s,
            pull_s=r.pull_s, run_s=r.run_s, refresh_s=r.refresh_s,
            read_s=r.read_s, total_s=r.total_s,
        )

    def make_bytes_rows(run, strategy, mode, phase, bytes_fetched) -> list[RefreshBytesRowMultimodal]:
        rows: list[RefreshBytesRowMultimodal] = []
        for op in OP_TYPES:
            for layer, b in sorted(bytes_fetched.get(op, {}).items()):
                rows.append(RefreshBytesRowMultimodal(
                    schema_version=MULTI_SCHEMA_VERSION, label=exp.label, base_image=exp.base_image,
                    num_models=len(names), models="|".join(names),
                    modified_model=exp.modified_model, mode=mode, run=run,
                    update_strategy=strategy, experiment_phase=phase,
                    layer=layer, op_type=op, bytes=b,
                ))
        return rows

    def mutate() -> Callable[[], None]:
        _mutate_model_files(modified_files)
        return lambda: _mutate_model_files(modified_files)  # xor is its own inverse

    def probe_disk_byte() -> int | None:
        with open(rep, "rb") as f:
            b = f.read(1)
        return b[0] if b else None

    return RefreshExperiment(
        label=exp.label, base_image=exp.base_image, name_slug=image_slug(exp.label),
        is_multi=True, groups=groups, in_paths=in_paths, size_str=_groups_size_str(groups),
        mutate=mutate,
        assert_mutated=lambda name, expected: _assert_first_byte_mutated(
            name, rep_in_path, orig_byte, expected,
        ),
        meta=meta, make_time_row=make_time_row, make_bytes_rows=make_bytes_rows,
        probe_disk_byte=probe_disk_byte,
    )


# ── output ───────────────────────────────────────────────────────────────


def print_results(time_rows: list) -> None:
    log.result(f"\n=== Refresh-vs-Baseline Results (mean ± stddev, n={N_RUNS} runs) ===")

    manual_lazy = [r for r in time_rows if r.update_strategy == "manual-lazy"]
    manual_oci = [r for r in time_rows if r.update_strategy == "manual-oci"]
    refresh = [r for r in time_rows if r.update_strategy == "refresh"]

    if manual_lazy:
        arr = np.array([(r.stop_total_s, r.pull_s, r.run_s, r.read_s) for r in manual_lazy], dtype=float)
        tot = np.array([r.total_s for r in manual_lazy])
        log.result(
            f"manual-lazy:  stop={arr[:,0].mean():.2f}±{arr[:,0].std():.2f}  "
            f"pull={arr[:,1].mean():.2f}±{arr[:,1].std():.2f}  "
            f"run={arr[:,2].mean():.2f}±{arr[:,2].std():.2f}  "
            f"read={arr[:,3].mean():.2f}±{arr[:,3].std():.2f}  "
            f"total={tot.mean():.2f}±{tot.std():.2f}"
        )
    if manual_oci:
        arr = np.array([(r.stop_total_s, r.pull_s, r.run_s, r.read_s) for r in manual_oci], dtype=float)
        tot = np.array([r.total_s for r in manual_oci])
        log.result(
            f"manual-oci:   stop={arr[:,0].mean():.2f}±{arr[:,0].std():.2f}  "
            f"pull={arr[:,1].mean():.2f}±{arr[:,1].std():.2f}  "
            f"run={arr[:,2].mean():.2f}±{arr[:,2].std():.2f}  "
            f"read={arr[:,3].mean():.2f}±{arr[:,3].std():.2f}  "
            f"total={tot.mean():.2f}±{tot.std():.2f}"
        )
    if refresh:
        arr = np.array([(r.refresh_s, r.read_s) for r in refresh], dtype=float)
        tot = np.array([r.total_s for r in refresh])
        log.result(
            f"refresh:   refresh={arr[:,0].mean():.2f}±{arr[:,0].std():.2f}  "
            f"read={arr[:,1].mean():.2f}±{arr[:,1].std():.2f}  "
            f"total={tot.mean():.2f}±{tot.std():.2f}"
        )


def _format_time_row(r) -> dict:
    row = asdict(r)
    for k, v in row.items():
        if isinstance(v, float):
            row[k] = f"{v:.4f}"
        elif v is None:
            row[k] = ""
    return row


def _write_time_rows(output_path: str, rows: list) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [f.name for f in fields(type(rows[0]))]
    write_csv(output_path, fieldnames, [_format_time_row(r) for r in rows])


def _write_bytes_rows(output_path: str, rows: list) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [f.name for f in fields(type(rows[0]))]
    write_csv(output_path, fieldnames, [asdict(r) for r in rows])


def save_results_csv(exp: RefreshExperiment, rows: list, execution_ts: str) -> None:
    if rows:
        _write_time_rows(refresh_csv_path(SCRIPT_DIR, exp.label, exp.base_image, execution_ts), rows)


def save_bytes_csv(exp: RefreshExperiment, rows: list, execution_ts: str) -> None:
    if rows:
        _write_bytes_rows(refresh_bytes_csv_path(SCRIPT_DIR, exp.label, exp.base_image, execution_ts), rows)


# ── chart ─────────────────────────────────────────────────────────────────


PHASE_COLORS = {
    "stop":    "#7f7f7f",
    "pull":    "#1f77b4",
    "run":     "#2ca02c",
    "refresh": "#9467bd",
    "read":    "#ff7f0e",
}


def plot(exp: RefreshExperiment, time_rows: list, execution_ts: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2))

    # (y, label, strategy, phase keys, per-row value extractor)
    arms = [
        (0, "manual update (lazy)", "manual-lazy", ("stop", "pull", "run", "read"),
         lambda r: (r.stop_total_s, r.pull_s, r.run_s, r.read_s)),
        (1, "manual update (OCI)", "manual-oci", ("stop", "pull", "run", "read"),
         lambda r: (r.stop_total_s, r.pull_s, r.run_s, r.read_s)),
        (2, "refresh", "refresh", ("refresh", "read"),
         lambda r: (r.refresh_s, r.read_s)),
    ]
    y_positions = [a[0] for a in arms]
    labels = [a[1] for a in arms]

    for y, _, strategy, keys, value_fn in arms:
        rows = [r for r in time_rows if r.update_strategy == strategy]
        if not rows:
            continue
        arr = np.array([value_fn(r) for r in rows], dtype=float)
        means = arr.mean(axis=0)
        cum_stds = np.cumsum(arr, axis=1).std(axis=0, ddof=0)
        left = 0.0
        for value, key in zip(means, keys):
            ax.barh(
                y, value, left=left, height=0.5,
                color=PHASE_COLORS[key], edgecolor=PHASE_COLORS[key], linewidth=0.5,
            )
            left += value
        ax.errorbar(
            np.cumsum(means), [y] * len(means), xerr=cum_stds,
            fmt="none", capsize=3, ecolor="black", elinewidth=1,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Elapsed time (s)")
    ax.set_title("model access time after config update")
    ax.invert_yaxis()
    ax.grid(True, linestyle="--", alpha=0.3, axis="x")

    handles = [
        mpatches.Patch(facecolor=c, edgecolor=c, label=lbl)
        for lbl, c in PHASE_COLORS.items()
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=5, fontsize=9, frameon=False,
    )

    figure_footer(fig, exp.label, exp.base_image, model_size=exp.size_str)
    output_path = refresh_chart_path(SCRIPT_DIR, exp.label, exp.base_image, execution_ts)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save_figure(fig, output_path)


def plot_bytes(exp: RefreshExperiment, bytes_rows: list, execution_ts: str) -> None:
    """One bar per (update_strategy, experiment_phase). Bytes = on_demand_bytes_fetched, mean across runs."""
    op = OP_TYPES[0]
    groups = [
        ("manual-lazy", "update", "manual update (lazy)"),
        ("manual-oci",  "update", "manual update (OCI)"),
        ("refresh",     "update", "refresh"),
    ]

    per_run_totals: dict[tuple[int, str, str], int] = {}
    for r in bytes_rows:
        if r.op_type != op:
            continue
        key = (r.run, r.update_strategy, r.experiment_phase)
        per_run_totals[key] = per_run_totals.get(key, 0) + r.bytes

    totals_by_key: dict[tuple[str, str], list[int]] = {}
    for (_, update_strategy, experiment_phase), total in per_run_totals.items():
        totals_by_key.setdefault((update_strategy, experiment_phase), []).append(total)

    if not any(totals_by_key.values()):
        log.info("plot_bytes: no data, skipping")
        return

    fig, ax = plt.subplots(figsize=(10, 4.0))
    bar_w = 0.6
    x_centers = np.arange(len(groups), dtype=float)

    means_gb: list[float] = []
    for group_index, (update_strategy, experiment_phase, _) in enumerate(groups):
        vals = totals_by_key.get((update_strategy, experiment_phase), [0])
        m_gb = float(np.mean(vals)) / (1024 ** 3)
        means_gb.append(m_gb)
        ax.bar(
            x_centers[group_index], m_gb, bar_w,
            color="#7f7f7f", edgecolor="white", linewidth=0.3,
        )
        if m_gb > 0:
            ax.text(
                x_centers[group_index], m_gb,
                _fmt_bytes(int(m_gb * (1024 ** 3))),
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

    if max(means_gb) > 0:
        ax.set_ylim(0, max(means_gb) * 1.18)

    ax.set_xticks(x_centers)
    ax.set_xticklabels([g[2] for g in groups], fontsize=10)

    ax.set_ylabel("GiB")
    ax.set_title("bytes fetched after config update", pad=24)
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")

    figure_footer(fig, exp.label, exp.base_image, fontsize=7, model_size=exp.size_str)
    output_path = refresh_bytes_chart_path(
        SCRIPT_DIR, exp.label, exp.base_image, execution_ts
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    save_figure(fig, output_path)


# ── main ───────────────────────────────────────────────────────────────


def _run_experiment(
    exp: RefreshExperiment, execution_ts: str,
) -> tuple[list, list]:
    """Two-phase build+measure for one prepared experiment: stargz phase
    (manual-lazy + refresh), then OCI phase (manual-oci)."""
    prepare_local_registry(exp.base_image, registry(CFG))
    clear_2dfs_cache(CFG)
    clear_registry(CFG, preserve_base=True, verbose=False)
    clear_artifacts(SCRIPT_DIR)

    artifacts_dir = refresh_artifacts_dir(
        SCRIPT_DIR, execution_ts, exp.label, exp.base_image, build_mode(LAZY_MODE)
    )
    toc_dir = os.path.join(artifacts_dir, "toc")
    log.result(f"TOC artifacts -> {toc_dir}")

    # ── phase 1: stargz — build, measure, then free local caches ──────
    log.result("=== Phase 1: stargz builds (manual-lazy + refresh) ===")
    os.makedirs(toc_dir, exist_ok=True)
    _build_mode_pair(exp, LAZY_MODE, artifacts_dir=artifacts_dir, toc_dir=toc_dir)

    time_rows, bytes_rows = measure_refresh(
        exp, execution_ts, ["manual-lazy", "refresh"]
    )

    log.result("=== Phase 1 cleanup: 2dfs cache + buildkit ===")
    clear_2dfs_cache(CFG)
    prune_buildkit()
    clear_artifacts(SCRIPT_DIR)

    # ── phase 2: OCI — build, measure ─────────────────────────────────
    log.result("=== Phase 2: OCI builds (manual-oci) ===")
    _build_mode_pair(exp, NO_LAZY_MODE)

    time_rows_2, bytes_rows_2 = measure_refresh(exp, execution_ts, ["manual-oci"])
    time_rows.extend(time_rows_2)
    bytes_rows.extend(bytes_rows_2)

    clear_registry(CFG, verbose=False, preserve_base=True)
    return time_rows, bytes_rows


def main(execution_ts: str | None = None) -> None:
    if execution_ts is None:
        execution_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_started = datetime.now(timezone.utc)
    log.set_verbose(VERBOSE)
    log.info(f"Mode: {LAZY_MODE}")
    log.info(f"N_RUNS: {N_RUNS}")

    ensure_buildkit()

    stargz_config_path = refresh_stargz_config_path(SCRIPT_DIR, execution_ts)
    os.makedirs(os.path.dirname(stargz_config_path), exist_ok=True)
    with open(stargz_config_path, "w") as f:
        f.write(read_base_config())
    log.result(f"Stargz config snapshot saved to {stargz_config_path}")

    all_single_time: list = []
    all_single_bytes: list = []
    all_multi_time: list = []
    all_multi_bytes: list = []
    experiments_meta: list[dict] = []

    for raw in EXPERIMENTS:
        prepared = prepare_multi(raw) if isinstance(raw, MultiModel) else prepare_single(raw)
        log.result(f"\n===== Experiment: {prepared.label} / {prepared.base_image} =====")
        experiments_meta.append(prepared.meta)

        time_rows, bytes_rows = _run_experiment(prepared, execution_ts)

        print_results(time_rows)
        save_results_csv(prepared, time_rows, execution_ts)
        save_bytes_csv(prepared, bytes_rows, execution_ts)
        plot(prepared, time_rows, execution_ts)
        plot_bytes(prepared, bytes_rows, execution_ts)

        if prepared.is_multi:
            all_multi_time.extend(time_rows)
            all_multi_bytes.extend(bytes_rows)
        else:
            all_single_time.extend(time_rows)
            all_single_bytes.extend(bytes_rows)

    if all_single_time:
        _write_time_rows(refresh_merged_csv_path(SCRIPT_DIR, execution_ts), all_single_time)
    if all_single_bytes:
        _write_bytes_rows(refresh_merged_bytes_csv_path(SCRIPT_DIR, execution_ts), all_single_bytes)
    if all_multi_time:
        _write_time_rows(refresh_multimodel_merged_csv_path(SCRIPT_DIR, execution_ts), all_multi_time)
    if all_multi_bytes:
        _write_bytes_rows(refresh_multimodel_merged_bytes_csv_path(SCRIPT_DIR, execution_ts), all_multi_bytes)

    write_run_json(
        refresh_run_metadata_path(SCRIPT_DIR, execution_ts),
        execution_ts=execution_ts,
        started_at=run_started,
        config=asdict(CFG),
        sections={
            "mode": LAZY_MODE,
            "oci_mode": NO_LAZY_MODE,
            "n_runs": N_RUNS,
            "update_strategies": ["manual-lazy", "manual-oci", "refresh"],
            "op_types": OP_TYPES,
            "prom_settle_s": PROM_SETTLE_S,
            "experiments": experiments_meta,
        },
    )

    clear_artifacts(SCRIPT_DIR)


if __name__ == "__main__":
    main()
