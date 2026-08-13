import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np

from shared import log
from shared.build_result import BuildResult
from build_performance.paths import (
    build_csv_path, build_chart_path,
    resource_csv_path, resource_chart_path,
    resource_cpu_charts_run_dir, resource_ram_charts_run_dir,
    resource_cores_charts_run_dir, resource_disk_charts_run_dir, resource_net_charts_run_dir,
    build_artifacts_dir,
    build_merged_csv_path, resource_merged_csv_path,
    build_run_metadata_path,
)
from shared.artifacts import snapshot_artifacts, clear_artifacts
from shared.config import load_config, build_tmpdir, data_volume
from shared.charts import MODE_COLORS, figure_footer, save_figure, write_csv
from shared.resource_monitor import ResourceMonitor, ResourceRow, derive_samples, write_resource_csv
from shared.resource_charts import plot_resource_aggregate, plot_resource_timeseries
from shared.registry import prepare_local_registry, registry
from shared.services import ensure_buildkit, prune_buildkit, clear_2dfs_cache
from shared.run_metadata import write_run_json
from build_performance import build_2dfs as b2
from build_performance import build_2dfs_stargz as b2s
from build_performance import build_2dfs_stargz_zstd as b2sz
from build_performance import build_stargz as bs
from build_performance import build_base as bb
from shared.packing import layers_for_percent
from shared import cv_splits as cv
from build_performance.prepare import (
    generate_build_artifacts, prepare_model_splits, print_packing_table,
    packing_preview_data, split_stats,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class SingleModel:
    """A single HuggingFace model split into allotments by capacity."""
    hf_model: str
    base_image: str


@dataclass(frozen=True)
class MultiModel:
    """Several pre-split CV models from splits/ packed into one image; each
    model is its own contiguous allotment block (stack order = list order)."""
    label: str
    split_dirs: list[str]
    base_image: str


EXPERIMENTS = [
    # SingleModel("openai-community/gpt2",        "docker.io/library/python:3.12-slim"),  # ~0.5GB
    # SingleModel("Qwen/Qwen2-1.5B",              "docker.io/library/python:3.12-slim"),  # ~3.09 GB
    # SingleModel("EleutherAI/pythia-1.4b", "docker.io/library/python:3.12-slim"),  # 3 GB
    # SingleModel("openlm-research/open_llama_3b", "docker.io/library/python:3.12-slim"),  # ~6.0 GB
    # SingleModel("Qwen/Qwen3.5-9B", "docker.io/ollama/ollama"),
    SingleModel("google/gemma-4-31B", "docker.io/ollama/ollama"),
    # MultiModel("cv-4model",
    #            ["resnet50_seperated", "deeplab_v3_seperated",
    #             "efficientnet_v2M_seperated", "yolov3_seperated"],
    #            "docker.io/tensorflow/tensorflow"),
]
CFG = load_config()
VERBOSE = False
MODES = ["2dfs", "2dfs-stargz", "2dfs-stargz-zstd", "stargz", "base"]
CAPACITIES = [0, 25, 50, 75, 100]
BUILD_SCHEMA_VERSION = 2

@dataclass(frozen=True)
class BuildRow:
    schema_version: int
    model: str
    base_image: str
    max_allowed_splits: int
    run: int
    capacity: int
    num_layers: int
    mode: str
    total_s: float
    pull_s: float
    build_s: float
    export_s: float


def _clear_cache(mode: str, cfg) -> None:
    if mode == "2dfs":
        b2.clear_cache(cfg)
    elif mode == "2dfs-stargz":
        b2s.clear_cache(cfg)
    elif mode == "2dfs-stargz-zstd":
        b2sz.clear_cache(cfg)
    elif mode == "stargz":
        bs.clear_cache()
    elif mode == "base":
        bb.clear_cache()
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _run_one(mode: str, n: int, cfg, source_image: str) -> BuildResult:
    if mode == "2dfs":
        return b2.run_one(n, cfg, source_image)
    elif mode == "2dfs-stargz":
        return b2s.run_one(n, cfg, source_image)
    elif mode == "2dfs-stargz-zstd":
        return b2sz.run_one(n, cfg, source_image)
    elif mode == "stargz":
        return bs.run_one(n, cfg)
    elif mode == "base":
        return bb.run_one(n, cfg)
    raise ValueError(f"Unknown mode: {mode}")


def measure_builds(
    model: str, base_image: str, max_allowed_splits: int,
    make_artifacts: Callable[[int], tuple[list[list[str]], int]], cfg=CFG,
    monitor: ResourceMonitor | None = None, execution_ts: str = "",
) -> list[BuildRow]:
    """make_artifacts(cap) writes 2dfs.json + Dockerfiles for that capacity and
    returns (groups, num_layers); it's the only per-arm (single vs multi) seam."""
    results: list[BuildRow] = []

    for run in range(cfg.build_n_runs):
        log.info(f"\n[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] === Run {run + 1}/{cfg.build_n_runs} ===")
        for cap in CAPACITIES:
            _, num_layers = make_artifacts(cap)
            log.info(f"\n[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] === Preparing capacity={cap}% ({num_layers} layer(s)) ===")
            if execution_ts:
                snapshot_artifacts(
                    SCRIPT_DIR,
                    build_artifacts_dir(SCRIPT_DIR, execution_ts, model, base_image, cap),
                )
            for i, mode in enumerate(MODES):
                if monitor:
                    monitor.set_context(mode, cap, run)
                log.info(f"\n[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] === {mode}: capacity={cap}% ({num_layers} layer(s)) ===")
                _clear_cache(mode, cfg)
                br = _run_one(mode, num_layers, cfg, base_image)
                if monitor:
                    monitor.set_idle()
                results.append(BuildRow(
                    schema_version=BUILD_SCHEMA_VERSION,
                    model=model,
                    base_image=base_image,
                    max_allowed_splits=max_allowed_splits,
                    run=run,
                    capacity=cap,
                    num_layers=num_layers,
                    mode=mode,
                    total_s=br.total_s,
                    pull_s=br.pull_s,
                    build_s=br.build_s,
                    export_s=br.export_s,
                ))

                is_last = (i == len(MODES) - 1) and (cap == CAPACITIES[-1]) and (run == cfg.build_n_runs - 1)
                if not is_last:
                    log.info(f"\nSleeping {cfg.build_cooldown}s before next...")
                    time.sleep(cfg.build_cooldown)

    return results


def _write_build_rows(output_path: str, results: list[BuildRow]) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [f.name for f in fields(BuildRow)]
    rows = [
        {
            **asdict(r),
            "total_s": f"{r.total_s:.4f}",
            "pull_s": f"{r.pull_s:.4f}",
            "build_s": f"{r.build_s:.4f}",
            "export_s": f"{r.export_s:.4f}",
        }
        for r in results
    ]
    write_csv(output_path, fieldnames, rows)


def save_csv(results: list[BuildRow], model: str, base_image: str, execution_ts: str) -> None:
    _write_build_rows(build_csv_path(SCRIPT_DIR, model, base_image, execution_ts), results)


def save_merged_csv(results: list[BuildRow], execution_ts: str) -> None:
    _write_build_rows(build_merged_csv_path(SCRIPT_DIR, execution_ts), results)


def plot(results: list[BuildRow], model: str, base_image: str, max_allowed_splits: int, execution_ts: str) -> None:
    capacities = sorted(set(r.capacity for r in results))

    fig, ax = plt.subplots(figsize=(max(8, len(capacities) * 2), 5))

    for mode in MODES:
        means = []
        stds = []
        for cap in capacities:
            vals = [r.total_s for r in results if r.mode == mode and r.capacity == cap]
            means.append(float(np.mean(vals)) if vals else float("nan"))
            stds.append(float(np.std(vals, ddof=0)) if vals else 0.0)
        ax.errorbar(capacities, means, yerr=stds, label=mode, color=MODE_COLORS[mode],
                    marker="o", capsize=3, linewidth=1.5)

    ax.set_xticks(capacities)
    ax.set_xlabel("Split capacity (%)")
    ax.set_ylabel("Total build time (s)")
    ax.set_title(f"Build performance (mean ± std, n={CFG.build_n_runs} runs)")
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    figure_footer(fig, model, base_image, max_allowed_splits=max_allowed_splits)
    path = build_chart_path(SCRIPT_DIR, model, base_image, execution_ts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_figure(fig, path)


def save_resource_csv(
    samples: list[ResourceRow], model: str, base_image: str, execution_ts: str,
) -> None:
    write_resource_csv(resource_csv_path(SCRIPT_DIR, model, base_image, execution_ts), samples, "capacity")


def save_merged_resource_csv(samples: list[ResourceRow], execution_ts: str) -> None:
    write_resource_csv(resource_merged_csv_path(SCRIPT_DIR, execution_ts), samples, "capacity")


@dataclass(frozen=True)
class Prepared:
    """Uniform handle the run loop consumes, regardless of experiment kind."""
    model: str
    base_image: str
    max_allowed_splits: int
    make_artifacts: Callable[[int], tuple[list[list[str]], int]]
    meta: dict


def prepare_single(exp: SingleModel) -> Prepared:
    chunks_dir, max_allowed_splits = prepare_model_splits(exp.hf_model)
    generate_build_artifacts(chunks_dir, max_allowed_splits, exp.base_image, CFG)
    num_layers_list = [layers_for_percent(max_allowed_splits, c) for c in CAPACITIES]
    labels = [f"{c}%" for c in CAPACITIES]
    print_packing_table(chunks_dir, exp.hf_model, max_allowed_splits, labels, num_layers_list)

    meta = {
        "model": exp.hf_model,
        "base_image": exp.base_image,
        "max_allowed_splits": max_allowed_splits,
        "splits": split_stats(chunks_dir),
        "packing_preview": packing_preview_data(chunks_dir, labels, num_layers_list, CAPACITIES),
    }

    def make_artifacts(cap: int) -> tuple[list[list[str]], int]:
        num_layers = layers_for_percent(max_allowed_splits, cap)
        groups = generate_build_artifacts(chunks_dir, num_layers, exp.base_image, CFG)
        return groups, num_layers

    return Prepared(exp.hf_model, exp.base_image, max_allowed_splits, make_artifacts, meta)


def prepare_multi(exp: MultiModel) -> Prepared:
    models = cv.prepare_cv_splits(exp.label, exp.split_dirs, SCRIPT_DIR)
    max_allowed_splits = sum(cv.full_columns(models))
    cv.print_cv_packing_table(exp.label, models, CAPACITIES)

    meta = {
        "model": exp.label,
        "base_image": exp.base_image,
        "max_allowed_splits": max_allowed_splits,
        "splits": cv.cv_split_stats(models),
        "packing_preview": cv.cv_packing_preview_data(models, CAPACITIES),
    }

    def make_artifacts(cap: int) -> tuple[list[list[str]], int]:
        groups, _ = cv.generate_cv_build_artifacts(
            models, cv.allocate_columns(models, cap), exp.base_image, CFG, SCRIPT_DIR,
        )
        return groups, len(groups)

    return Prepared(exp.label, exp.base_image, max_allowed_splits, make_artifacts, meta)


def main():
    log.set_verbose(VERBOSE)
    ensure_buildkit()
    prune_buildkit()
    clear_2dfs_cache(CFG)
    execution_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_started = datetime.now(timezone.utc)

    all_results: list[BuildRow] = []
    all_samples: list[ResourceRow] = []
    experiments_meta: list[dict] = []
    for exp in EXPERIMENTS:
        prepared = prepare_multi(exp) if isinstance(exp, MultiModel) else prepare_single(exp)
        model, base_image = prepared.model, prepared.base_image
        max_allowed_splits = prepared.max_allowed_splits
        log.result(f"\n===== Experiment: {model} / {base_image} (max_allowed_splits={max_allowed_splits}) =====")

        experiments_meta.append(prepared.meta)

        prepare_local_registry(base_image, registry(CFG))

        monitor = None
        if CFG.build_with_resource:
            monitor = ResourceMonitor(model, base_image, max_allowed_splits, build_tmpdir(CFG),
                                      data_volume(CFG), registry(CFG))
            monitor.start()

        results = measure_builds(model, base_image, max_allowed_splits, prepared.make_artifacts, CFG, monitor=monitor, execution_ts=execution_ts)

        if monitor:
            samples = monitor.stop()
            save_resource_csv(samples, model, base_image, execution_ts)  # raw counters
            enriched = derive_samples(samples)
            plot_resource_aggregate(
                enriched, modes=MODES, n_runs=CFG.build_n_runs,
                xlabel="Split capacity (%)",
                title=f"Resource usage during builds (mean ± std, n={CFG.build_n_runs} runs)",
                model=model, base_image=base_image, max_allowed_splits=max_allowed_splits,
                output_path=resource_chart_path(SCRIPT_DIR, model, base_image, execution_ts),
            )
            plot_resource_timeseries(
                enriched, model=model, base_image=base_image,
                max_allowed_splits=max_allowed_splits,
                dimension_label="capacity", dimension_tag="cap",
                cpu_dir=resource_cpu_charts_run_dir(SCRIPT_DIR, execution_ts),
                ram_dir=resource_ram_charts_run_dir(SCRIPT_DIR, execution_ts),
                cores_dir=resource_cores_charts_run_dir(SCRIPT_DIR, execution_ts),
                disk_dir=resource_disk_charts_run_dir(SCRIPT_DIR, execution_ts),
                net_dir=resource_net_charts_run_dir(SCRIPT_DIR, execution_ts),
            )
            all_samples.extend(samples)

        save_csv(results, model, base_image, execution_ts)
        plot(results, model, base_image, max_allowed_splits, execution_ts)
        all_results.extend(results)

    if all_results:
        save_merged_csv(all_results, execution_ts)
    if all_samples:
        save_merged_resource_csv(all_samples, execution_ts)

    write_run_json(
        build_run_metadata_path(SCRIPT_DIR, execution_ts),
        execution_ts=execution_ts,
        started_at=run_started,
        config=asdict(CFG),
        sections={
            "modes": MODES,
            "capacities": CAPACITIES,
            "experiments": experiments_meta,
        },
    )

    clear_artifacts(SCRIPT_DIR)


if __name__ == "__main__":
    main()
