import os
import subprocess
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from shared import log
from shared import cv_splits as cv
from shared.charts import MODE_COLORS, figure_footer, save_figure, write_csv
from pull_performance.paths import (
    pull_csv_path, pull_chart_path, pull_artifacts_dir, pull_stargz_config_path,
    pull_merged_csv_path, pull_run_metadata_path,
    pull_multimodel_csv_path, pull_multimodel_chart_path,
    pull_multimodel_merged_csv_path, pull_multimodel_resource_merged_csv_path,
    pull_resource_csv_path, pull_resource_merged_csv_path, pull_resource_chart_path,
    pull_resource_cpu_charts_run_dir, pull_resource_ram_charts_run_dir,
    pull_resource_cores_charts_run_dir, pull_resource_disk_charts_run_dir,
    pull_resource_net_charts_run_dir,
)
from shared.config import load_config, build_tmpdir, data_volume
from shared.resource_monitor import ResourceMonitor, ResourceRow, derive_samples, write_resource_csv
from shared.resource_charts import plot_resource_aggregate, plot_resource_timeseries
from shared.run_metadata import write_run_json
from shared.registry import prepare_local_registry, clear_registry, registry
from shared.services import ensure_buildkit, clear_stargz_cache, prune_buildkit
from shared.artifacts import clear_artifacts
from shared.model import cleanup_pull_experiment
from shared.stargz_config import read_base_config
from pull_performance.prepare import (
    prepare_model_splits,
    build_and_push_2dfs, build_and_push_2dfs_stargz, build_and_push_2dfs_stargz_zstd,
    build_and_push_stargz, build_and_push_base,
)
from shared.services import clear_2dfs_cache
from shared.packing import layers_for_percent
from pull_performance.images import (
    pull_name_2dfs, pull_name_2dfs_stargz, pull_name_2dfs_stargz_zstd,
    pull_name_stargz, pull_name_base,
)


@dataclass(frozen=True)
class SingleModel:
    """A single HuggingFace model split into allotments; x-axis = partition %."""
    hf_model: str
    base_image: str


@dataclass(frozen=True)
class MultiModel:
    """Several pre-split CV models packed one-allotment-per-model into one image;
    x-axis = how many models are accessed (cumulative, stack order = list order)."""
    label: str
    split_dirs: list[str]
    base_image: str


EXPERIMENTS = [
    # SingleModel("openai-community/gpt2", "docker.io/library/python:3.12-slim"),  # ~0.5GB  ~50 MB
    # SingleModel("Qwen/Qwen2-1.5B", "docker.io/library/python:3.12-slim"),        # ~3.09 GB
    # SingleModel("openlm-research/open_llama_3b", "docker.io/ollama/ollama"),     # ~6.0 GB
    # SingleModel("Qwen/Qwen3.5-9B", "docker.io/ollama/ollama"),
    MultiModel("cv-4model",
               ["resnet50_seperated", "deeplab_v3_seperated",
                "efficientnet_v2M_seperated", "yolov3_seperated"],
               "docker.io/tensorflow/tensorflow"),
]
CFG = load_config()
VERBOSE = False
MODES = ["2dfs", "2dfs-stargz", "2dfs-stargz-zstd", "stargz", "base"]
PARTITION_PERCENTS = [25, 50, 75, 100]
SCHEMA_VERSION = 1
MULTI_SCHEMA_VERSION = 1

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class PullRow:
    schema_version: int
    model: str
    base_image: str
    max_allowed_splits: int
    run: int
    partition_pct: int
    num_splits: int
    mode: str
    pull_s: float
    run_s: float
    total_s: float


@dataclass(frozen=True)
class MultiModelPullRow:
    schema_version: int
    label: str
    base_image: str
    num_models_total: int
    run: int
    num_models: int        # k = x position (cumulative count accessed)
    models: str            # ordered prefix names, e.g. "resnet50|deeplab_v3"
    num_allotments: int
    mode: str
    pull_s: float
    run_s: float
    total_s: float


@dataclass(frozen=True)
class ExperimentPlan:
    """What prepare_single/prepare_multi resolve an experiment into, and what the
    measure loop consumes — uniform across both arms. steps = [(x_value, n_allotments), ...];
    make_row builds the arm's row; is_multi selects the output/plot path."""
    model: str
    base_image: str
    allotments: list[list[str]]
    max_allowed_splits: int
    steps: list[tuple[int, int]]
    dim_col: str
    x_label: str
    x_tag: str
    is_multi: bool
    meta: dict
    make_row: Callable[[str, int, int, int, float, float], object]


# ── helpers ────────────────────────────────────────────────────────


def _next_container_name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _timed_pull(cmd: list[str]) -> float:
    start = time.perf_counter()
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return time.perf_counter() - start


def _timed_run(cmd: list[str]) -> float:
    start = time.perf_counter()
    subprocess.run(cmd, check=True, capture_output=not log.VERBOSE)
    return time.perf_counter() - start


def _run_cmd(allotments: list[list[str]], n: int) -> list[str]:
    files = " ".join(
        f"/{os.path.basename(p)}"
        for a in allotments[:n] for p in a
    )
    log.info(f"  reading files: {files}")
    return ["sh", "-c", f"cat {files} > /dev/null"]


# ── pull functions ─────────────────────────────────────────────────


def pull_base(source_image: str, cfg, num_splits: int) -> float:
    image = pull_name_base(source_image, cfg, num_splits)
    log.info(f"Pulling base image: {image}")
    elapsed = _timed_pull(["sudo", "ctr", "images", "pull", "--plain-http", image])
    log.result(f"  base pull ({num_splits} splits): {elapsed:.2f}s")
    return elapsed


def pull_stargz(source_image: str, cfg) -> float:
    image = pull_name_stargz(source_image, cfg)
    log.info(f"Pulling stargz image: {image}")
    elapsed = _timed_pull(["sudo", "ctr-remote", "images", "rpull", "--plain-http", image])
    log.result(f"  stargz pull: {elapsed:.2f}s")
    return elapsed


def pull_2dfs(source_image: str, cfg, num_allotments: int) -> float:
    image = pull_name_2dfs(source_image, cfg, num_allotments)
    log.info(f"Pulling 2dfs ({num_allotments} allotments): {image}")
    elapsed = _timed_pull(["sudo", "ctr", "images", "pull", "--plain-http", image])
    log.result(f"  2dfs pull ({num_allotments} allotments): {elapsed:.2f}s")
    return elapsed


def pull_2dfs_stargz(source_image: str, cfg, num_allotments: int) -> float:
    image = pull_name_2dfs_stargz(source_image, cfg, num_allotments)
    log.info(f"Pulling 2dfs-stargz ({num_allotments} allotments): {image}")
    elapsed = _timed_pull(["sudo", "ctr-remote", "images", "rpull", "--plain-http", image])
    log.result(f"  2dfs-stargz pull ({num_allotments} allotments): {elapsed:.2f}s")
    return elapsed


def pull_2dfs_stargz_zstd(source_image: str, cfg, num_allotments: int) -> float:
    image = pull_name_2dfs_stargz_zstd(source_image, cfg, num_allotments)
    log.info(f"Pulling 2dfs-stargz-zstd ({num_allotments} allotments): {image}")
    elapsed = _timed_pull(["sudo", "ctr-remote", "images", "rpull", "--plain-http", image])
    log.result(f"  2dfs-stargz-zstd pull ({num_allotments} allotments): {elapsed:.2f}s")
    return elapsed


# ── run functions ──────────────────────────────────────────────────


def run_base(image: str, allotments: list[list[str]], n: int) -> float:
    name = _next_container_name("run-base")
    log.info(f"Running base container: {name} (reading {n} allotments)")
    elapsed = _timed_run([
        "sudo", "ctr", "run", "--rm", image, name, *_run_cmd(allotments, n),
    ])
    log.result(f"  base run: {elapsed:.2f}s")
    return elapsed


def run_stargz(image: str, allotments: list[list[str]], n: int) -> float:
    name = _next_container_name("run-stargz")
    log.info(f"Running stargz container: {name} (reading {n} allotments)")
    elapsed = _timed_run([
        "sudo", "ctr-remote", "run", "--rm", "--snapshotter=stargz",
        image, name, *_run_cmd(allotments, n),
    ])
    log.result(f"  stargz run: {elapsed:.2f}s")
    return elapsed


def run_2dfs(image: str, allotments: list[list[str]], n: int) -> float:
    name = _next_container_name("run-2dfs")
    log.info(f"Running 2dfs container: {name} (reading {n} allotments)")
    elapsed = _timed_run([
        "sudo", "ctr", "run", "--rm", image, name, *_run_cmd(allotments, n),
    ])
    log.result(f"  2dfs run: {elapsed:.2f}s")
    return elapsed


def run_2dfs_stargz(image: str, allotments: list[list[str]], n: int) -> float:
    name = _next_container_name("run-2dfs-stargz")
    log.info(f"Running 2dfs-stargz container: {name} (reading {n} allotments)")
    elapsed = _timed_run([
        "sudo", "ctr-remote", "run", "--rm", "--snapshotter=stargz",
        image, name, *_run_cmd(allotments, n),
    ])
    log.result(f"  2dfs-stargz run: {elapsed:.2f}s")
    return elapsed


def run_2dfs_stargz_zstd(image: str, allotments: list[list[str]], n: int) -> float:
    name = _next_container_name("run-2dfs-stargz-zstd")
    log.info(f"Running 2dfs-stargz-zstd container: {name} (reading {n} allotments)")
    elapsed = _timed_run([
        "sudo", "ctr-remote", "run", "--rm", "--snapshotter=stargz",
        image, name, *_run_cmd(allotments, n),
    ])
    log.result(f"  2dfs-stargz-zstd run: {elapsed:.2f}s")
    return elapsed


# ── orchestration ──────────────────────────────────────────────────


def _prepare_mode(
    mode: str, allotments: list[list[str]], base_splits: list[int],
    source_image: str, cfg, model: str, execution_ts: str,
) -> None:
    def art(n: int | None = None) -> str:
        return pull_artifacts_dir(SCRIPT_DIR, execution_ts, model, source_image, mode, n)
    if mode == "base":
        build_and_push_base(allotments, base_splits, source_image, cfg, artifacts_dir_fn=art)
    elif mode == "stargz":
        build_and_push_stargz(allotments, source_image, cfg, artifacts_dir=art())
    elif mode == "2dfs":
        clear_2dfs_cache(cfg)
        build_and_push_2dfs(allotments, source_image, cfg, artifacts_dir=art())
    elif mode == "2dfs-stargz":
        clear_2dfs_cache(cfg)
        build_and_push_2dfs_stargz(allotments, source_image, cfg, artifacts_dir=art())
    elif mode == "2dfs-stargz-zstd":
        clear_2dfs_cache(cfg)
        build_and_push_2dfs_stargz_zstd(allotments, source_image, cfg, artifacts_dir=art())
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _measure_one(mode: str, allotments: list[list[str]], n: int, source_image: str, cfg) -> tuple[int, float, float]:
    if mode == "base":
        pull_t = pull_base(source_image, cfg, n)
        run_t = run_base(pull_name_base(source_image, cfg, n), allotments, n)
    elif mode == "stargz":
        pull_t = pull_stargz(source_image, cfg)
        run_t = run_stargz(pull_name_stargz(source_image, cfg), allotments, n)
    elif mode == "2dfs":
        pull_t = pull_2dfs(source_image, cfg, n)
        run_t = run_2dfs(pull_name_2dfs(source_image, cfg, n), allotments, n)
    elif mode == "2dfs-stargz":
        pull_t = pull_2dfs_stargz(source_image, cfg, n)
        run_t = run_2dfs_stargz(pull_name_2dfs_stargz(source_image, cfg, n), allotments, n)
    elif mode == "2dfs-stargz-zstd":
        pull_t = pull_2dfs_stargz_zstd(source_image, cfg, n)
        run_t = run_2dfs_stargz_zstd(pull_name_2dfs_stargz_zstd(source_image, cfg, n), allotments, n)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return (n, pull_t, run_t)


def split_stats(allotments: list[list[str]]) -> dict:
    """Aggregate split-pool stats (sizes in MB) for run metadata."""
    sizes = [
        os.path.getsize(p)
        for a in allotments for p in a if p.endswith(".safetensors")
    ]
    return {
        "num_safetensors": len(sizes),
        "total_mb": round(sum(sizes) / (1024 ** 2), 1),
        "max_file_mb": round(max(sizes) / (1024 ** 2), 1) if sizes else 0.0,
    }


def packing_preview_data(allotments: list[list[str]], max_allowed_splits: int) -> list[dict]:
    """Structured per-partition-pct: #allotments read + their sizes (MB).

    Cumulative: each pct reads the first n allotments."""
    out: list[dict] = []
    for pct in PARTITION_PERCENTS:
        n = layers_for_percent(max_allowed_splits, pct)
        sizes_mb = [
            round(sum(os.path.getsize(p) for p in a) / (1024 ** 2), 1)
            for a in allotments[:n]
        ]
        out.append({
            "partition_pct": pct,
            "num_splits": n,
            "allotment_sizes_mb": sizes_mb,
        })
    return out


def print_packing_table(allotments: list[list[str]], model: str, max_allowed_splits: int) -> None:
    """Print per-partition-pct (#allotments read, allotment sizes in MB)."""
    log.result(
        f"\n=== Packing preview: {model} "
        f"(max_allowed_splits={max_allowed_splits}) ==="
    )
    log.result(f"{'pct':>10}  {'allotments':>11}  sizes (MB)")
    log.result("-" * 60)
    for e in packing_preview_data(allotments, max_allowed_splits):
        sizes_str = "[" + ", ".join(f"{s:.1f}" for s in e["allotment_sizes_mb"]) + "]"
        log.result(f"{e['partition_pct']:>9}%  {e['num_splits']:>11}  {sizes_str}")


def prepare_single(exp: SingleModel) -> ExperimentPlan:
    allotments, max_allowed_splits = prepare_model_splits(exp.hf_model)
    print_packing_table(allotments, exp.hf_model, max_allowed_splits)
    steps = [(pct, layers_for_percent(max_allowed_splits, pct)) for pct in PARTITION_PERCENTS]
    meta = {
        "model": exp.hf_model,
        "base_image": exp.base_image,
        "max_allowed_splits": max_allowed_splits,
        "splits": split_stats(allotments),
        "packing_preview": packing_preview_data(allotments, max_allowed_splits),
    }

    def make_row(mode: str, run: int, x_value: int, n: int, pull_t: float, run_t: float) -> PullRow:
        return PullRow(
            schema_version=SCHEMA_VERSION, model=exp.hf_model, base_image=exp.base_image,
            max_allowed_splits=max_allowed_splits, run=run, partition_pct=x_value,
            num_splits=n, mode=mode, pull_s=pull_t, run_s=run_t, total_s=pull_t + run_t,
        )

    return ExperimentPlan(exp.hf_model, exp.base_image, allotments, max_allowed_splits,
                          steps, "partition_pct", "Partition size (%)", "pct", False, meta, make_row)


def prepare_multi(exp: MultiModel) -> ExperimentPlan:
    models = cv.prepare_cv_splits(exp.label, exp.split_dirs, SCRIPT_DIR)
    groups, _ = cv.pack_cv(models, [1] * len(models))  # one allotment per model
    names = [m.name for m in models]
    n_models = len(models)
    cv.print_cv_packing_table(exp.label, models, [100])
    steps = [(k, k) for k in range(1, n_models + 1)]  # 1 allotment/model → n == k
    meta = {
        "model": exp.label,
        "base_image": exp.base_image,
        "max_allowed_splits": n_models,
        "models": names,
        "splits": cv.cv_split_stats(models),
    }

    def make_row(mode: str, run: int, x_value: int, n: int, pull_t: float, run_t: float) -> MultiModelPullRow:
        return MultiModelPullRow(
            schema_version=MULTI_SCHEMA_VERSION, label=exp.label, base_image=exp.base_image,
            num_models_total=n_models, run=run, num_models=x_value,
            models="|".join(names[:x_value]), num_allotments=n, mode=mode,
            pull_s=pull_t, run_s=run_t, total_s=pull_t + run_t,
        )

    return ExperimentPlan(exp.label, exp.base_image, groups, n_models,
                          steps, "num_models", "Models accessed", "model", True, meta, make_row)


def measure(
    allotments: list[list[str]], source_image: str, cfg, model: str, execution_ts: str,
    steps: list[tuple[int, int]], dim_col: str,
    make_row: Callable[[str, int, int, int, float, float], object],
    monitor: ResourceMonitor | None = None,
) -> list:
    """Build+push each mode once, then for every (x_value, n) step pull the first
    n allotments and read them. steps come from the ExperimentPlan: x_value is
    partition % (single) or models accessed (multi). make_row(mode, run, x_value,
    n, pull, run) builds the arm's row type; dim_col is the resource-monitor
    dimension name."""
    results: list = []

    base_splits = [n for _, n in steps]

    clear_registry(cfg, preserve_base=True, verbose=False)
    for mode in MODES:
        log.info(f"\n=== Preparing mode: {mode} ===")
        prepare_local_registry(source_image, registry(cfg))
        _prepare_mode(mode, allotments, base_splits, source_image, cfg, model, execution_ts)
        clear_2dfs_cache(cfg)
        prune_buildkit()

        for run in range(CFG.pull_n_runs):
            log.info(f"\n[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] === Run {run + 1}/{CFG.pull_n_runs} ===")
            for x_value, n in steps:
                ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                log.info(f"\n[{ts}] === {mode}: {dim_col}={x_value} ({n} allotments) ===")
                clear_stargz_cache()
                if monitor:
                    monitor.set_context(mode, x_value, run)
                _, pull_t, run_t = _measure_one(mode, allotments, n, source_image, cfg)
                if monitor:
                    monitor.set_idle()
                results.append(make_row(mode, run, x_value, n, pull_t, run_t))
                log.info(f"\nSleeping {cfg.pull_cooldown}s before next...")
                time.sleep(cfg.pull_cooldown)

        clear_registry(cfg, preserve_base=True)

    return results


# ── output: shared ─────────────────────────────────────────────────


def _write_rows(output_path: str, results: list) -> None:
    """Write any PullRow/MultiModelPullRow list to CSV (fields from the row type)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [f.name for f in fields(type(results[0]))]
    rows = [{
        **asdict(r),
        "pull_s": f"{r.pull_s:.4f}",
        "run_s": f"{r.run_s:.4f}",
        "total_s": f"{r.total_s:.4f}",
    } for r in results]
    write_csv(output_path, fieldnames, rows)


def _plot_pull_run(
    results: list, dim_attr: str, x_label: str, xtick_fmt: Callable[[int], str],
    title: str, output_path: str, footer_label: str, base_image: str,
    footer_extra: str | None = None,
) -> None:
    """Stacked pull(hatched)+run bars per mode across the x-axis values found in
    `dim_attr` (partition_pct or num_models)."""
    xs = sorted({getattr(r, dim_attr) for r in results})
    x = np.arange(len(xs))
    n_modes = len(MODES)
    width = min(0.8 / n_modes, 0.15)

    fig, ax = plt.subplots(figsize=(max(10, n_modes * 2), 6))

    for i, mode in enumerate(MODES):
        color = MODE_COLORS[mode]
        offset = (i - (n_modes - 1) / 2) * width
        mean_pulls, std_pulls, mean_runs, std_totals = [], [], [], []
        for xv in xs:
            group = [(r.pull_s, r.run_s) for r in results
                     if r.mode == mode and getattr(r, dim_attr) == xv]
            if group:
                pull_arr = np.array([g[0] for g in group])
                run_arr = np.array([g[1] for g in group])
                mean_pulls.append(float(pull_arr.mean()))
                std_pulls.append(float(pull_arr.std(ddof=0)))
                mean_runs.append(float(run_arr.mean()))
                std_totals.append(float((pull_arr + run_arr).std(ddof=0)))
            else:
                mean_pulls.append(0.0)
                std_pulls.append(0.0)
                mean_runs.append(0.0)
                std_totals.append(0.0)
        ax.bar(x + offset, mean_pulls, width, yerr=std_pulls, capsize=3,
               color=color, alpha=0.5, hatch="//", edgecolor=color, linewidth=0.5)
        ax.bar(x + offset, mean_runs, width, bottom=mean_pulls, yerr=std_totals, capsize=3,
               color=color, edgecolor=color, linewidth=0.5, label=mode)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Time (s)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([xtick_fmt(v) for v in xs])
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")

    method_handles = [mpatches.Patch(facecolor=MODE_COLORS[m], edgecolor=MODE_COLORS[m], label=m)
                      for m in MODES]
    pull_patch = mpatches.Patch(facecolor="gray", alpha=0.5, hatch="//",
                                edgecolor="gray", label="pull")
    run_patch = mpatches.Patch(facecolor="gray", edgecolor="gray", label="run")
    ax.legend(handles=method_handles + [pull_patch, run_patch], loc="upper left")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    figure_footer(fig, footer_label, base_image, extra=footer_extra)
    save_figure(fig, output_path)


# ── output: single-model ───────────────────────────────────────────


def print_results(results: list[PullRow]) -> None:
    pcts = sorted({r.partition_pct for r in results})
    col = 26
    header_modes = "  ".join(f"{m:>{col}}" for m in MODES)
    subheader = "  ".join(f"{'pull(m±s) run(m±s)':>{col}}" for _ in MODES)
    log.result(f"\n=== Pull + Run Performance Results (mean ± stddev, n={CFG.pull_n_runs} runs) ===")
    log.result(f"{'pct':>8}  {header_modes}")
    log.result(f"{'':>8}  {subheader}")
    log.result("-" * (10 + (col + 2) * len(MODES)))
    for pct in pcts:
        def fmt(mode: str) -> str:
            group = [(r.pull_s, r.run_s) for r in results if r.mode == mode and r.partition_pct == pct]
            if not group:
                return "N/A"
            pull_arr = np.array([g[0] for g in group])
            run_arr = np.array([g[1] for g in group])
            return (
                f"{pull_arr.mean():.1f}±{pull_arr.std(ddof=0):.1f} "
                f"{run_arr.mean():.1f}±{run_arr.std(ddof=0):.1f}"
            )
        row = "  ".join(f"{fmt(m):>{col}}" for m in MODES)
        log.result(f"{pct:>7}%  {row}")


def save_csv(results: list[PullRow], model: str, base_image: str, execution_ts: str) -> None:
    pcts = sorted({r.partition_pct for r in results})
    _write_rows(pull_csv_path(SCRIPT_DIR, model, base_image, len(pcts), execution_ts), results)


def save_merged_csv(results: list[PullRow], execution_ts: str) -> None:
    _write_rows(pull_merged_csv_path(SCRIPT_DIR, execution_ts), results)


def plot(results: list[PullRow], model: str, base_image: str, execution_ts: str) -> None:
    pcts = sorted({r.partition_pct for r in results})
    _plot_pull_run(
        results, "partition_pct", "Partition size (%)", lambda p: f"{p}%",
        f"Pull + Run Performance (mean ± stddev, n={CFG.pull_n_runs} runs)",
        pull_chart_path(SCRIPT_DIR, model, base_image, len(pcts), execution_ts),
        model, base_image,
    )


# ── output: multimodel ─────────────────────────────────────────────


def save_multimodel_csv(results: list[MultiModelPullRow], label: str, base_image: str, execution_ts: str) -> None:
    n = len(sorted({r.num_models for r in results}))
    _write_rows(pull_multimodel_csv_path(SCRIPT_DIR, label, base_image, n, execution_ts), results)


def save_merged_multimodel_csv(results: list[MultiModelPullRow], execution_ts: str) -> None:
    _write_rows(pull_multimodel_merged_csv_path(SCRIPT_DIR, execution_ts), results)


def plot_multimodel(results: list[MultiModelPullRow], label: str, base_image: str, execution_ts: str) -> None:
    ks = sorted({r.num_models for r in results})
    full_order = max(results, key=lambda r: r.num_models).models.split("|")
    access_order = " -> ".join(name.removesuffix("_seperated") for name in full_order)
    _plot_pull_run(
        results, "num_models", "Models accessed", str,
        f"Multimodel pull + run (mean ± stddev, n={CFG.pull_n_runs} runs)",
        pull_multimodel_chart_path(SCRIPT_DIR, label, base_image, len(ks), execution_ts),
        label, base_image, footer_extra=f"access order: {access_order}",
    )


# ── output: resource ───────────────────────────────────────────────


def save_resource_csv(samples: list[ResourceRow], model: str, base_image: str, execution_ts: str, dim_col: str) -> None:
    write_resource_csv(pull_resource_csv_path(SCRIPT_DIR, model, base_image, execution_ts), samples, dim_col)


def save_merged_resource_csv(samples: list[ResourceRow], execution_ts: str) -> None:
    write_resource_csv(pull_resource_merged_csv_path(SCRIPT_DIR, execution_ts), samples, "partition_pct")


def save_merged_multimodel_resource_csv(samples: list[ResourceRow], execution_ts: str) -> None:
    write_resource_csv(pull_multimodel_resource_merged_csv_path(SCRIPT_DIR, execution_ts), samples, "num_models")


# ── main ───────────────────────────────────────────────────────────


def main():
    log.set_verbose(VERBOSE)
    clear_artifacts(SCRIPT_DIR)
    ensure_buildkit()
    execution_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_started = datetime.now(timezone.utc)
    log.info(f"Modes: {MODES}")
    log.info(f"Partition percents: {PARTITION_PERCENTS}")
    log.info(f"Runs: {CFG.pull_n_runs}")

    log.info("Pre-run cleanup...")
    for exp in EXPERIMENTS:
        name = exp.label if isinstance(exp, MultiModel) else exp.hf_model
        cleanup_pull_experiment(name, SCRIPT_DIR, CFG)

    stargz_config_path = pull_stargz_config_path(SCRIPT_DIR, execution_ts)
    os.makedirs(os.path.dirname(stargz_config_path), exist_ok=True)
    with open(stargz_config_path, "w") as f:
        f.write(read_base_config())
    log.result(f"Stargz config snapshot saved to {stargz_config_path}")

    all_single: list[PullRow] = []
    all_multi: list[MultiModelPullRow] = []
    single_samples: list[ResourceRow] = []
    multi_samples: list[ResourceRow] = []
    experiments_meta: list[dict] = []
    for exp in EXPERIMENTS:
        prepared = prepare_multi(exp) if isinstance(exp, MultiModel) else prepare_single(exp)
        model, base_image = prepared.model, prepared.base_image
        max_allowed_splits = prepared.max_allowed_splits
        log.result(f"\n===== Experiment: {model} / {base_image} (max_splits={max_allowed_splits}) =====")

        experiments_meta.append(prepared.meta)

        monitor = ResourceMonitor(model, base_image, max_allowed_splits, build_tmpdir(CFG),
                                  data_volume(CFG), registry(CFG))
        monitor.start()

        results = measure(prepared.allotments, base_image, CFG, model, execution_ts,
                          prepared.steps, prepared.dim_col, prepared.make_row, monitor=monitor)

        samples = monitor.stop()
        save_resource_csv(samples, model, base_image, execution_ts, prepared.dim_col)  # raw counters
        enriched = derive_samples(samples)
        plot_resource_aggregate(
            enriched, modes=MODES, n_runs=CFG.pull_n_runs,
            xlabel=prepared.x_label,
            title=f"Resource usage during pull+run (mean ± std, n={CFG.pull_n_runs} runs)",
            model=model, base_image=base_image, max_allowed_splits=max_allowed_splits,
            output_path=pull_resource_chart_path(SCRIPT_DIR, model, base_image, execution_ts),
        )
        plot_resource_timeseries(
            enriched, model=model, base_image=base_image,
            max_allowed_splits=max_allowed_splits,
            dimension_label="models" if prepared.is_multi else "partition",
            dimension_tag=prepared.x_tag,
            dimension_unit="" if prepared.is_multi else "%",
            cpu_dir=pull_resource_cpu_charts_run_dir(SCRIPT_DIR, execution_ts),
            ram_dir=pull_resource_ram_charts_run_dir(SCRIPT_DIR, execution_ts),
            cores_dir=pull_resource_cores_charts_run_dir(SCRIPT_DIR, execution_ts),
            disk_dir=pull_resource_disk_charts_run_dir(SCRIPT_DIR, execution_ts),
            net_dir=pull_resource_net_charts_run_dir(SCRIPT_DIR, execution_ts),
        )

        if prepared.is_multi:
            save_multimodel_csv(results, model, base_image, execution_ts)
            plot_multimodel(results, model, base_image, execution_ts)
            all_multi.extend(results)
            multi_samples.extend(samples)
        else:
            print_results(results)
            save_csv(results, model, base_image, execution_ts)
            plot(results, model, base_image, execution_ts)
            all_single.extend(results)
            single_samples.extend(samples)
        cleanup_pull_experiment(model, SCRIPT_DIR, CFG)

    if all_single:
        save_merged_csv(all_single, execution_ts)
    if all_multi:
        save_merged_multimodel_csv(all_multi, execution_ts)
    if single_samples:
        save_merged_resource_csv(single_samples, execution_ts)
    if multi_samples:
        save_merged_multimodel_resource_csv(multi_samples, execution_ts)

    write_run_json(
        pull_run_metadata_path(SCRIPT_DIR, execution_ts),
        execution_ts=execution_ts,
        started_at=run_started,
        config=asdict(CFG),
        sections={
            "modes": MODES,
            "partition_percents": PARTITION_PERCENTS,
            "experiments": experiments_meta,
        },
    )

    clear_artifacts(SCRIPT_DIR)


if __name__ == "__main__":
    main()
