import os
import time
from collections.abc import Callable
from dataclasses import asdict, fields
from datetime import datetime, timezone
from typing import TypeVar

import matplotlib.pyplot as plt
import numpy as np

from shared import log
from shared.build_result import BuildResult
from shared.charts import MODE_COLORS, figure_footer, save_figure, write_csv
from build_performance import build_2dfs as b2
from build_performance import build_2dfs_stargz as b2s
from build_performance import build_2dfs_stargz_zstd as b2sz
from build_performance import build_base as bb
from build_performance import build_stargz as bs

DIRECTIONS = ["top_to_bottom", "bottom_to_top"]
_DIRECTION_TITLES = {"top_to_bottom": "Top to Bottom", "bottom_to_top": "Bottom to Top"}

# The rebuild axis value (mutation % for single-model, # models for multi-model).
AxisT = TypeVar("AxisT")


def make_methods(modes: list[str], cfg, base_image: str) -> list[tuple]:
    """(name, build_fn(n), clear_fn) for each requested mode, in MODES order."""
    all_methods = [
        ("2dfs",             lambda n, bi=base_image: b2.build_only(n, cfg, bi),   lambda: b2.clear_cache(cfg)),
        ("2dfs-stargz",      lambda n, bi=base_image: b2s.build_only(n, cfg, bi),  lambda: b2s.clear_cache(cfg)),
        ("2dfs-stargz-zstd", lambda n, bi=base_image: b2sz.build_only(n, cfg, bi), lambda: b2sz.clear_cache(cfg)),
        ("stargz",           lambda n: bs.build_only(n, cfg),                      lambda: bs.clear_cache()),
        ("base",             lambda n: bb.build_only(n, cfg),                      lambda: bb.clear_cache()),
    ]
    return [(name, bf, cf) for name, bf, cf in all_methods if name in modes]


def get_buckets_to_mutate(items: list, n: int, direction: str) -> list:
    """Top/bottom n of an ordered list. top_to_bottom -> items[-n:], bottom_to_top -> items[:n]."""
    if direction == "top_to_bottom":
        return items[-n:]
    return items[:n]


def mutate_buckets(buckets: list[list[str]], ext: str, mutate_fn: Callable[[str], None]) -> list[str]:
    """Mutate every file ending in `ext` across the picked buckets; return the
    flat path list so the caller can restore with a second mutate pass."""
    mutated: list[str] = []
    for bucket in buckets:
        for path in bucket:
            if path.endswith(ext):
                mutate_fn(path)
                mutated.append(path)
    return mutated


def measure_rebuild_matrix(
    *, n_runs: int, axis_values: list[AxisT], directions: list[str], methods: list[tuple],
    max_allowed_splits: int,
    select_targets: Callable[[str, AxisT], tuple[int, list[list[str]]]],
    mutate_ext: str, mutate_fn: Callable[[str], None],
    make_row: Callable[[int, AxisT, str, str, int, BuildResult], object],
    cooldown: float, axis_label: str,
) -> list:
    """Run the runs × axis × direction × mode rebuild matrix.

    Per cell: clear cache, full cold build, mutate the selected buckets, time the
    incremental rebuild, then restore the mutated files. select_targets(direction,
    value) -> (n, buckets); make_row(run, value, direction, mode, n, br) -> Row.
    """
    results: list = []
    for run in range(n_runs):
        log.info(f"\n[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] === Run {run + 1}/{n_runs} ===")
        for value in axis_values:
            for direction in directions:
                n, target_buckets = select_targets(direction, value)
                for mode_name, build_fn, clear_fn in methods:
                    log.info(f"\n[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] "
                             f"=== {axis_label}={value} (n={n}), {direction}, {mode_name} ===")
                    t0 = time.time()
                    clear_fn()
                    log.info(f"Cache clear took {time.time() - t0:.2f}s")
                    build_fn(max_allowed_splits)

                    mutated = mutate_buckets(target_buckets, mutate_ext, mutate_fn)
                    try:
                        br: BuildResult = build_fn(max_allowed_splits)
                    finally:
                        for path in mutated:
                            mutate_fn(path)

                    results.append(make_row(run, value, direction, mode_name, n, br))
                    log.result(f"Total time: {br.total_s:.2f}s")
                    log.info(f"\nSleeping {cooldown}s before next...")
                    time.sleep(cooldown)
    return results


def write_rebuild_csv(output_path: str, results: list) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [f.name for f in fields(results[0])] if results else []
    rows = [{**asdict(r), "total_s": f"{r.total_s:.4f}"} for r in results]
    write_csv(output_path, fieldnames, rows)


def plot_rebuild(
    results: list, *, modes: list[str], axis_values: list, axis_attr: str,
    xlabel: str, n_runs: int, model: str, base_image: str, max_allowed_splits: int,
    suptitle: str, output_path: str,
) -> None:
    """Two panels (top_to_bottom / bottom_to_top), one line per mode, x = axis_values."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, direction in zip(axes, DIRECTIONS):
        for mode in modes:
            means, stds = [], []
            for value in axis_values:
                vals = [
                    r.total_s for r in results
                    if r.direction == direction and r.mode == mode and getattr(r, axis_attr) == value
                ]
                means.append(float(np.mean(vals)) if vals else float("nan"))
                stds.append(float(np.std(vals, ddof=0)) if vals else 0.0)
            ax.errorbar(axis_values, means, yerr=stds, label=mode, color=MODE_COLORS[mode],
                        marker="o", capsize=3, linewidth=1.5)
        ax.set_xticks(axis_values)
        ax.set_xlabel(xlabel)
        ax.set_title(_DIRECTION_TITLES[direction])
        ax.legend(fontsize="small")
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[0].set_ylabel("Total rebuild time (s)")
    fig.suptitle(suptitle)
    fig.tight_layout()
    figure_footer(fig, model, base_image, max_allowed_splits=max_allowed_splits)
    save_figure(fig, output_path)
