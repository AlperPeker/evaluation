import os
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from shared import log
from shared.charts import MODE_COLORS, figure_footer, add_run_dots, bar_group_xticks, save_figure
from pull_performance.paths import (
    stargz_config_charts_run_dir, stargz_config_csv_path, stargz_config_chart_path, stargz_config_log_path,
    stargz_config_merged_csv_path, stargz_config_run_metadata_path, stargz_config_base_config_path,
    stargz_config_artifacts_dir,
)
from shared.artifacts import clear_artifacts
from shared.config import load_config
from shared.registry import prepare_local_registry, clear_registry, registry, image_slug
from shared.run_metadata import write_run_json
from shared.stargz_config import read_base_config, apply_overrides, apply_stargz_config
from pull_performance.measure import _timed_pull, _timed_run, _run_cmd, _write_rows
from shared.services import clear_stargz_cache, clear_2dfs_cache, save_stargz_run_log
from shared.packing import layers_for_percent
from pull_performance.prepare import (
    prepare_model_splits, build_and_push_2dfs_stargz, build_and_push_2dfs_stargz_zstd,
)
from shared.model import cleanup_pull_experiment
from pull_performance.images import pull_name_2dfs_stargz, pull_name_2dfs_stargz_zstd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

EXPERIMENTS = [
    ("openlm-research/open_llama_3b", "docker.io/ollama/ollama"), 
]
MODES = ["2dfs-stargz-zstd"]
CONFIG_OPTIONS: list[tuple[dict, str]] = [
    ({"fuse.passthrough": True}, "with passthrough"),
    ({"fuse.passthrough": False}, "no passthrough"),
]
PARTITION_PERCENTS = [25, 50, 75, 100]
N_RUNS = 3
CFG = load_config()
VERBOSE = False
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StargzConfigRow:
    schema_version: int
    model: str
    base_image: str
    mode: str
    config_label: str
    run: int
    partition_pct: int
    num_splits: int
    max_allowed_splits: int
    pull_s: float
    run_s: float
    total_s: float


# ── image naming ───────────────────────────────────────────────────


def _pull_name(mode: str, source_image: str, cfg, n: int) -> str:
    if mode == "2dfs-stargz":
        return pull_name_2dfs_stargz(source_image, cfg, n)
    elif mode == "2dfs-stargz-zstd":
        return pull_name_2dfs_stargz_zstd(source_image, cfg, n)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ── prepare ────────────────────────────────────────────────────────


def _prepare_mode(mode: str, allotments: list[list[str]], source_image: str, cfg, artifacts_dir: str | None = None) -> None:
    clear_2dfs_cache(cfg)
    if mode == "2dfs-stargz":
        build_and_push_2dfs_stargz(allotments, source_image, cfg, artifacts_dir)
    elif mode == "2dfs-stargz-zstd":
        build_and_push_2dfs_stargz_zstd(allotments, source_image, cfg, artifacts_dir)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ── measure ────────────────────────────────────────────────────────


def _measure_config_option(
    mode: str, allotments: list[list[str]], max_allowed_splits: int, source_image: str, cfg,
    config_label: str, run_idx: int, model: str, base_image: str, execution_ts: str,
) -> list[tuple[int, int, float, float]]:
    results = []
    for pct in PARTITION_PERCENTS:
        n = layers_for_percent(max_allowed_splits, pct)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        log.info(f"\n[{ts}] === {mode}: {pct}% ({n} allotments) ===")
        clear_stargz_cache()

        image = _pull_name(mode, source_image, cfg, n)
        pull_start_s = time.time()
        pull_t = _timed_pull(["sudo", "ctr-remote", "images", "rpull", "--plain-http", "--use-containerd-labels", image])
        log.result(f"  pull: {pull_t:.2f}s")

        name = f"run-stargz-cfg-{uuid.uuid4().hex[:8]}"
        run_t = _timed_run([
            "sudo", "ctr-remote", "run", "--rm", "--snapshotter=stargz",
            image, name, *_run_cmd(allotments, n),
        ])
        run_end_s = time.time()
        log.result(f"  run: {run_t:.2f}s")

        save_stargz_run_log(pull_start_s, run_end_s, stargz_config_log_path(SCRIPT_DIR, model, base_image, mode, config_label, n, run_idx, execution_ts))

        results.append((pct, n, pull_t, run_t))
        log.info(f"\nSleeping {cfg.pull_cooldown}s before next...")
        time.sleep(cfg.pull_cooldown)
    return results


# ── orchestration ──────────────────────────────────────────────────


def measure(
    allotments: list[list[str]], max_allowed_splits: int, source_image: str, cfg,
    model: str, base_image: str, execution_ts: str,
) -> list[StargzConfigRow]:
    results: list[StargzConfigRow] = []

    base_config = read_base_config()

    def _prepare_all_images():
        log.info("\n=== Preparing images ===")
        for mode in MODES:
            log.info(f"\n--- Preparing mode: {mode} ---")
            prepare_local_registry(source_image, registry(cfg))
            artifacts_dir = stargz_config_artifacts_dir(SCRIPT_DIR, execution_ts, model, base_image, mode)
            _prepare_mode(mode, allotments, source_image, cfg, artifacts_dir)

    try:
        for overrides, label in CONFIG_OPTIONS:
            log.info(f"\n=== Config option: {label} ===")
            clear_registry(cfg, preserve_base=True)
            _prepare_all_images()
            config_content = apply_overrides(base_config, overrides)
            apply_stargz_config(config_content)

            for mode in MODES:
                for run in range(N_RUNS):
                    log.info(f"\n[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] "
                             f"=== Run {run + 1}/{N_RUNS} | {mode} | {label} ===")
                    for pct, n, pull_t, run_t in _measure_config_option(
                        mode, allotments, max_allowed_splits, source_image, cfg,
                        label, run, model, base_image, execution_ts,
                    ):
                        results.append(StargzConfigRow(
                            schema_version=SCHEMA_VERSION, model=model, base_image=base_image,
                            mode=mode, config_label=label, run=run, partition_pct=pct,
                            num_splits=n, max_allowed_splits=max_allowed_splits,
                            pull_s=pull_t, run_s=run_t, total_s=pull_t + run_t,
                        ))
    finally:
        log.info("\n=== Restoring base stargz config ===")
        apply_stargz_config(base_config)

    return results


# ── output ─────────────────────────────────────────────────────────


def save_csv(results: list[StargzConfigRow], model: str, base_image: str, execution_ts: str) -> None:
    _write_rows(stargz_config_csv_path(SCRIPT_DIR, model, base_image, execution_ts), results)


def save_merged_csv(results: list[StargzConfigRow], execution_ts: str) -> None:
    _write_rows(stargz_config_merged_csv_path(SCRIPT_DIR, execution_ts), results)


def plot(results: list[StargzConfigRow], model: str, base_image: str, execution_ts: str) -> None:
    os.makedirs(stargz_config_charts_run_dir(SCRIPT_DIR, execution_ts), exist_ok=True)

    config_labels = [label for _, label in CONFIG_OPTIONS]
    pcts = sorted({r.partition_pct for r in results})
    n_configs = len(config_labels)
    width = min(0.8 / n_configs, 0.15)

    for mode in MODES:
        color = MODE_COLORS[mode]
        fig, ax = plt.subplots(figsize=(max(10, n_configs * 2), 6))
        x = np.arange(len(pcts))

        for i, label in enumerate(config_labels):
            entries = [r for r in results if r.mode == mode and r.config_label == label]
            offset = (i - (n_configs - 1) / 2) * width
            med_pulls = []
            med_runs = []
            for j, pct in enumerate(pcts):
                group = [(r.pull_s, r.run_s) for r in entries if r.partition_pct == pct]
                med_p = float(np.median([g[0] for g in group])) if group else 0.0
                med_r = float(np.median([g[1] for g in group])) if group else 0.0
                med_pulls.append(med_p)
                med_runs.append(med_r)
                x_center = x[j] + offset + width / 2
                add_run_dots(ax, x_center, [g[0] + g[1] for g in group])

            # vary lightness per config option so bars are distinguishable
            alpha = 0.4 + 0.6 * (i / max(n_configs - 1, 1))
            ax.bar(x + offset, med_pulls, width, color=color, alpha=alpha * 0.6,
                   hatch="//", edgecolor=color, linewidth=0.5)
            ax.bar(x + offset, med_runs, width, bottom=med_pulls, color=color,
                   alpha=alpha, edgecolor=color, linewidth=0.5, label=label)

        bar_group_xticks(ax, len(pcts), n_configs, width, [f"{p}%" for p in pcts])
        ax.set_xlabel("Partition size (%)")
        ax.set_ylabel("Time (s)")
        ax.set_title(
            f"Pull + Run by stargz config ({mode}, "
            f"median, n={N_RUNS} runs, dots = individual runs)"
        )
        ax.grid(True, linestyle="--", alpha=0.3, axis="y")

        pull_patch = mpatches.Patch(facecolor="gray", alpha=0.5, hatch="//",
                                    edgecolor="gray", label="pull")
        run_patch = mpatches.Patch(facecolor="gray", edgecolor="gray", label="run")
        config_handles = [
            mpatches.Patch(facecolor=color,
                           alpha=0.4 + 0.6 * (i / max(n_configs - 1, 1)),
                           edgecolor=color, label=label)
            for i, label in enumerate(config_labels)
        ]
        ax.legend(handles=config_handles + [pull_patch, run_patch], loc="upper left")

        figure_footer(fig, model, base_image, lower_in=0.25)
        fig.tight_layout()

        output_path = stargz_config_chart_path(SCRIPT_DIR, model, base_image, mode, execution_ts)
        save_figure(fig, output_path)


# ── main ───────────────────────────────────────────────────────────


def main():
    log.set_verbose(VERBOSE)
    clear_artifacts(SCRIPT_DIR)
    execution_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_started = datetime.now(timezone.utc)
    config_labels = [label for _, label in CONFIG_OPTIONS]
    log.info(f"Modes: {MODES}")
    log.info(f"Config options: {config_labels}")
    log.info(f"Partition percents: {PARTITION_PERCENTS}")
    log.info(f"Runs: {N_RUNS}")

    log.info("Pre-run cleanup...")
    for model, _ in EXPERIMENTS:
        cleanup_pull_experiment(model, SCRIPT_DIR, CFG)

    base_config_path = stargz_config_base_config_path(SCRIPT_DIR, execution_ts)
    os.makedirs(os.path.dirname(base_config_path), exist_ok=True)
    with open(base_config_path, "w") as f:
        f.write(read_base_config())
    log.result(f"Stargz base config snapshot saved to {base_config_path}")

    all_results: list[StargzConfigRow] = []
    experiments_meta: list[dict] = []
    for model, base_image in EXPERIMENTS:
        allotments, max_allowed_splits = prepare_model_splits(model)
        log.result(f"\n===== Experiment: {model} / {base_image} (max_splits={max_allowed_splits}) =====")
        prepare_local_registry(base_image, registry(CFG))

        results = measure(allotments, max_allowed_splits, base_image, CFG, model, base_image, execution_ts)

        save_csv(results, model, base_image, execution_ts)
        plot(results, model, base_image, execution_ts)
        all_results.extend(results)
        experiments_meta.append({
            "model": model,
            "base_image": base_image,
            "max_allowed_splits": max_allowed_splits,
            "partition_percents": PARTITION_PERCENTS,
        })
        cleanup_pull_experiment(model, SCRIPT_DIR, CFG)

    if all_results:
        save_merged_csv(all_results, execution_ts)

    write_run_json(
        stargz_config_run_metadata_path(SCRIPT_DIR, execution_ts),
        execution_ts=execution_ts,
        started_at=run_started,
        config=asdict(CFG),
        sections={
            "modes": MODES,
            "config_options": [{"label": label, "overrides": overrides} for overrides, label in CONFIG_OPTIONS],
            "partition_percents": PARTITION_PERCENTS,
            "n_runs": N_RUNS,
            "experiments": experiments_meta,
        },
    )

    clear_artifacts(SCRIPT_DIR)


if __name__ == "__main__":
    main()
