import os
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from shared import log
from shared.charts import figure_footer, save_figure, write_csv
from shared.config import load_config
from shared.registry import prepare_local_registry, clear_registry, registry
from shared.services import clear_stargz_cache, save_stargz_run_log, collect_stargz_journal_since
from shared.stargz_config import read_base_config, apply_overrides, apply_stargz_config
from shared.model import cleanup_pull_experiment
from shared.services import clear_2dfs_cache
from shared.packing import layers_for_percent
from pull_performance.measure import _timed_pull, _timed_run, _run_cmd
from pull_performance.paths import (
    prefetch_stages_charts_run_dir, prefetch_stages_csv_path, prefetch_stages_chart_path,
    prefetch_stages_log_path, prefetch_stages_artifacts_dir,
    prefetch_stages_merged_csv_path, prefetch_stages_run_metadata_path, prefetch_stages_base_config_path,
)
from shared.run_metadata import write_run_json
from shared.artifacts import clear_artifacts
from pull_performance.prepare import (
    prepare_model_splits, build_and_push_2dfs_stargz, build_and_push_2dfs_stargz_zstd,
)
from pull_performance.prefetch_common import (
    poll_until_prefetch_done, prefetch_span, bg_fetch_spans, passthrough_open_spans,
    pull_name,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

EXPERIMENTS = [
    ("Qwen/Qwen3.5-9B", "docker.io/ollama/ollama"),
]
MODES = ["2dfs-stargz-zstd"]
# 8 -> 25/50/75/100% = 2/4/6/8 allotments
MAX_ALLOTMENTS = 8
PARTITION_PERCENTS = [25, 50, 75, 100]
N_RUNS = 1
SCHEMA_VERSION = 1
# request_timeout_sec is important otherwise the layer fetch inside prefetch fails 
# as it is done in a single fetch call and times out with default config
# prefetch_chunk_size might also be relevant. the default (0) makes layer be fetched in a single call
CONFIG_OPTIONS: list[tuple[dict, str]] = [
    ({"noprefetch": True, "no_background_fetch": True, "log_file_access": True}, "no prefetch"),
    ({"noprefetch": False, "prefetch_async_size": 0, "no_background_fetch": True, "log_file_access": True, "prefetch_timeout_sec": 120, "resolver.request_timeout_sec": 120}, "prefetch"),
    ({"noprefetch": False, "prefetch_async_size": 1, "no_background_fetch": True, "log_file_access": True, "prefetch_timeout_sec": 120, "resolver.request_timeout_sec": 120}, "prefetch, async"),
    ({"noprefetch": False, "prefetch_async_size": 1, "no_background_fetch": False, "log_file_access": True, "prefetch_timeout_sec": 120, "resolver.request_timeout_sec": 120}, "prefetch, async, bg fetch"),
]
CFG = load_config()
VERBOSE = False

PULL_COLOR = "#ffd966"
PREFETCH_COLOR = "#9467bd"
BG_DOWNLOAD_COLOR = "#d5d5d5"
FILE_OPEN_ON_DEMAND_COLOR = "#d62728"
FILE_OPEN_CACHE_COLOR = "#17becf"
CREATE_COLOR = "#a9cce3"
RUN_COLOR = "#f5b7b1"
LAYER_CMAP = plt.get_cmap("tab10")  # per-layer color (shared by prefetch + file_open)


# ── data structures ────────────────────────────────────────────────

@dataclass(frozen=True)
class PrefetchStageRow:
    schema_version: int
    model: str
    base_image: str
    run: int
    mode: str
    config: str
    partition_pct: int
    n_allotments: int
    max_allowed_splits: int
    pull_rel_start_s: str
    pull_rel_end_s: str
    prefetch_rel_start_s: str
    prefetch_rel_end_s: str
    prefetch_rel_events: str
    bg_download_rel_start_s: str
    bg_download_rel_end_s: str
    file_open_cache_rel_events: str
    file_open_on_demand_rel_events: str
    create_rel_start_s: str
    create_rel_end_s: str
    task_rel_start_s: str
    task_rel_end_s: str

@dataclass
class PrefetchStageSpan:
    """In-memory span with absolute timestamps + nested event lists; used for plotting."""
    run: int
    mode: str
    partition_pct: int
    n_allotments: int
    max_allowed_splits: int
    config_label: str
    pull_start_s: float
    pull_end_s: float
    prefetch_start_s: float | None
    prefetch_end_s: float | None
    prefetch_layer_events: list[tuple[str, float, float]]
    bg_download_start_s: float | None
    bg_download_end_s: float | None
    file_open_cache_spans: list[tuple[str, float, float]]
    file_open_on_demand_spans: list[tuple[str, float, float]]
    create_start_s: float
    create_end_s: float
    task_start_s: float
    task_end_s: float


# ── prepare ────────────────────────────────────────────────────────


def _prepare_mode(mode: str, allotments: list[list[str]], source_image: str, cfg, artifacts_dir: str | None = None) -> None:
    clear_2dfs_cache(cfg)
    if mode == "2dfs-stargz":
        build_and_push_2dfs_stargz(allotments, source_image, cfg, artifacts_dir)
    elif mode == "2dfs-stargz-zstd":
        build_and_push_2dfs_stargz_zstd(allotments, source_image, cfg, artifacts_dir)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _prepare_all_images(
    source_image: str, cfg, allotments: list[list[str]],
    model: str | None = None, execution_ts: str | None = None,
) -> None:
    log.info("\n=== Preparing images ===")
    for mode in MODES:
        log.info(f"\n--- Preparing mode: {mode} ---")
        prepare_local_registry(source_image, registry(cfg))
        artifacts_dir = (
            prefetch_stages_artifacts_dir(SCRIPT_DIR, execution_ts, model, source_image, mode)
            if model and execution_ts else None
        )
        _prepare_mode(mode, allotments, source_image, cfg, artifacts_dir)


# ── measure ────────────────────────────────────────────────────────


def _measure_config_option(
    mode: str, allotments: list[list[str]], max_allowed_splits: int, source_image: str, cfg,
    config_label: str, run_idx: int, model: str, base_image: str, execution_ts: str,
) -> list[PrefetchStageSpan]:
    results = []
    for pct in PARTITION_PERCENTS:
        n = layers_for_percent(max_allowed_splits, pct)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        log.info(f"\n[{ts}] === {mode}: {pct}% ({n} allotments) ===")
        clear_stargz_cache()

        image = pull_name(mode, source_image, cfg, n)
        pull_start_s = time.time()
        pull_t = _timed_pull(["sudo", "ctr-remote", "images", "rpull", "--plain-http", "--use-containerd-labels", image])
        pull_end_s = pull_start_s + pull_t
        log.result(f"  pull: {pull_t:.2f}s")

        name = f"run-stargz-stages-{uuid.uuid4().hex[:8]}"
        create_start_s = time.time()
        create_t = _timed_run([
            "sudo", "ctr-remote", "c", "create", "--snapshotter=stargz",
            "--", image, name, *_run_cmd(allotments, n),
        ])
        create_end_s = create_start_s + create_t
        log.result(f"  create: {create_t:.2f}s")

        task_start_s = time.time()
        task_t = _timed_run(["sudo", "ctr-remote", "t", "start", name])
        task_end_s = task_start_s + task_t
        log.result(f"  task: {task_t:.2f}s")

        subprocess.run(
            ["sudo", "ctr-remote", "t", "kill", "-s", "9", name],
            check=False, capture_output=True,
        )
        subprocess.run(
            ["sudo", "ctr-remote", "c", "rm", name],
            check=True, capture_output=not log.VERBOSE,
        )

        # Wait for any background prefetch to settle, then parse the journal.
        events = poll_until_prefetch_done(pull_start_s)
        span = prefetch_span(events)
        if span:
            log.result(f"  prefetch span: {span[0] - pull_start_s:.2f}s → {span[1] - pull_start_s:.2f}s ({len(events)} layers)")
        else:
            log.result(f"  prefetch span: none")

        raw_entries = collect_stargz_journal_since(pull_start_s)
        dl_span = bg_fetch_spans(raw_entries)
        if dl_span:
            log.result(f"  bg download span: {dl_span[0] - pull_start_s:.2f}s → {dl_span[1] - pull_start_s:.2f}s")
        file_open_cache_spans, file_open_on_demand_spans = passthrough_open_spans(raw_entries)
        log.result(f"  file_open cache: {len(file_open_cache_spans)}, on-demand: {len(file_open_on_demand_spans)}")
        ends = [task_end_s]
        if span:
            ends.append(span[1])
        if dl_span:
            ends.append(dl_span[1])
        if file_open_cache_spans:
            ends.append(max(e for _, _, e in file_open_cache_spans))
        if file_open_on_demand_spans:
            ends.append(max(e for _, _, e in file_open_on_demand_spans))
        save_stargz_run_log(
            pull_start_s,
            max(ends),
            prefetch_stages_log_path(SCRIPT_DIR, model, base_image, mode, config_label, n, run_idx, execution_ts),
        )

        results.append(PrefetchStageSpan(
            run=run_idx,
            mode=mode,
            partition_pct=pct,
            n_allotments=n,
            max_allowed_splits=max_allowed_splits,
            config_label=config_label,
            pull_start_s=pull_start_s,
            pull_end_s=pull_end_s,
            prefetch_start_s=span[0] if span else None,
            prefetch_end_s=span[1] if span else None,
            prefetch_layer_events=[(e.layer_sha, e.start_s, e.end_s) for e in events],
            bg_download_start_s=dl_span[0] if dl_span else None,
            bg_download_end_s=dl_span[1] if dl_span else None,
            file_open_cache_spans=file_open_cache_spans,
            file_open_on_demand_spans=file_open_on_demand_spans,
            create_start_s=create_start_s,
            create_end_s=create_end_s,
            task_start_s=task_start_s,
            task_end_s=task_end_s,
        ))
        log.info(f"\nSleeping {cfg.pull_cooldown}s before next...")
        time.sleep(cfg.pull_cooldown)
    return results


# ── orchestration ──────────────────────────────────────────────────


def measure(
    allotments: list[list[str]], max_allowed_splits: int, source_image: str, cfg,
    model: str, base_image: str, execution_ts: str,
) -> dict[tuple[str, str], list[PrefetchStageSpan]]:
    results: dict[tuple[str, str], list[PrefetchStageSpan]] = {}

    base_config = read_base_config()

    try:
        # TODO: the built image is byte-identical across config options (only the
        # client-side stargz config changes). Hoist
        # clear_registry + _prepare_all_images out of this loop to build/push once
        # instead of rebuilding per config option.
        for overrides, label in CONFIG_OPTIONS:
            log.info(f"\n=== Config option: {label} ===")
            clear_registry(cfg, preserve_base=True)
            _prepare_all_images(source_image, cfg, allotments, model, execution_ts)
            config_content = apply_overrides(base_config, overrides)
            apply_stargz_config(config_content)

            for mode in MODES:
                key = (mode, label)
                results[key] = []
                for run in range(N_RUNS):
                    log.info(f"\n[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] "
                             f"=== Run {run + 1}/{N_RUNS} | {mode} | {label} ===")
                    results[key].extend(_measure_config_option(
                        mode, allotments, max_allowed_splits, source_image, cfg,
                        label, run, model, base_image, execution_ts,
                    ))
    finally:
        log.info("\n=== Restoring base stargz config ===")
        apply_stargz_config(base_config)

    return results


# ── output ─────────────────────────────────────────────────────────


def _rel(t: float | None, ref: float) -> str:
    return f"{t - ref:.3f}" if t is not None else ""


def _encode_events(spans: list[tuple[str, float, float]], ref: float) -> str:
    return "|".join(f"{sha[:12]}:{a - ref:.3f}:{b - ref:.3f}" for sha, a, b in spans)


def _build_rows(
    results: dict[tuple[str, str], list[PrefetchStageSpan]], model: str, base_image: str,
) -> list[PrefetchStageRow]:
    rows = []
    for (mode, label), entries in results.items():
        for s in entries:
            ref = s.pull_start_s
            rows.append(PrefetchStageRow(
                schema_version=SCHEMA_VERSION, model=model, base_image=base_image,
                run=s.run, mode=mode, config=label, partition_pct=s.partition_pct,
                n_allotments=s.n_allotments, max_allowed_splits=s.max_allowed_splits,
                pull_rel_start_s="0.000", pull_rel_end_s=_rel(s.pull_end_s, ref),
                prefetch_rel_start_s=_rel(s.prefetch_start_s, ref),
                prefetch_rel_end_s=_rel(s.prefetch_end_s, ref),
                prefetch_rel_events=_encode_events(s.prefetch_layer_events, ref),
                bg_download_rel_start_s=_rel(s.bg_download_start_s, ref),
                bg_download_rel_end_s=_rel(s.bg_download_end_s, ref),
                file_open_cache_rel_events=_encode_events(s.file_open_cache_spans, ref),
                file_open_on_demand_rel_events=_encode_events(s.file_open_on_demand_spans, ref),
                create_rel_start_s=_rel(s.create_start_s, ref), create_rel_end_s=_rel(s.create_end_s, ref),
                task_rel_start_s=_rel(s.task_start_s, ref), task_rel_end_s=_rel(s.task_end_s, ref),
            ))
    return rows


def _write_rows(output_path: str, rows: list[PrefetchStageRow]) -> None:
    """Write PrefetchStageRow list to CSV; columns derive from the dataclass fields."""
    fieldnames = [f.name for f in fields(PrefetchStageRow)]
    write_csv(output_path, fieldnames, [asdict(r) for r in rows])


def save_csv(rows: list[PrefetchStageRow], model: str, base_image: str, execution_ts: str) -> None:
    if not rows:
        log.info("No data to save.")
        return
    _write_rows(prefetch_stages_csv_path(SCRIPT_DIR, model, base_image, execution_ts), rows)


def save_merged_csv(rows: list[PrefetchStageRow], execution_ts: str) -> None:
    if not rows:
        return
    _write_rows(prefetch_stages_merged_csv_path(SCRIPT_DIR, execution_ts), rows)


def _median_span(entries: list[PrefetchStageSpan], pct: int) -> PrefetchStageSpan | None:
    """Pick the run with median total elapsed time at this partition percentage."""
    runs = [s for s in entries if s.partition_pct == pct]
    if not runs:
        return None
    runs.sort(key=lambda s: s.task_end_s - s.pull_start_s)
    return runs[len(runs) // 2]


def plot(
    results: dict[tuple[str, str], list[PrefetchStageSpan]],
    model: str,
    base_image: str,
    execution_ts: str,
) -> None:
    os.makedirs(prefetch_stages_charts_run_dir(SCRIPT_DIR, execution_ts), exist_ok=True)

    config_labels = [label for _, label in CONFIG_OPTIONS]
    pcts = sorted({s.partition_pct for entries in results.values() for s in entries})

    bar_h = 0.14
    prefetch_slot_h = 5 * bar_h  # per-layer micro-lanes for prefetch (up to 10 layers)
    # Slot stack (top→bottom): prefetch(5), file_open(1), bg(1), pull+run(1) → total 8*bar_h
    offsets = {
        "prefetch": -1.5 * bar_h,
        "file_open": 1.5 * bar_h,
        "bg_download": 2.5 * bar_h,
        "run": 3.5 * bar_h,
    }
    offsets["pull"] = offsets["run"]
    row_spacing = 8 * bar_h + 0.3  # row stack height + gap

    for mode in MODES:
        n_cols = len(config_labels)
        fig, axes = plt.subplots(
            1, n_cols,
            figsize=(max(14, 4 * n_cols), max(3.5, row_spacing * len(pcts) + 1.5)),
            squeeze=False,
            sharey=True,
        )

        x_max = 0.0
        for col_idx, label in enumerate(config_labels):
            ax = axes[0][col_idx]
            entries = results.get((mode, label), [])

            for y_idx, pct in enumerate(pcts):
                y = y_idx * row_spacing
                s = _median_span(entries, pct)
                if s is None:
                    continue
                ref = s.pull_start_s

                pull_left = 0.0
                pull_w = s.pull_end_s - ref
                ax.barh(y + offsets["pull"], pull_w, left=pull_left, height=bar_h * 0.9,
                        color=PULL_COLOR, edgecolor=PULL_COLOR)

                # Build per-layer color map from union of prefetch + file_open events,
                # ordered by first-seen time. Same color is used for that layer in both
                # the prefetch micro-lane and the file_open lane.
                first_seen: dict[str, float] = {}
                for sha, start, _ in s.prefetch_layer_events:
                    if sha and (sha not in first_seen or start < first_seen[sha]):
                        first_seen[sha] = start
                for sha, start, _ in s.file_open_cache_spans:
                    if sha and (sha not in first_seen or start < first_seen[sha]):
                        first_seen[sha] = start
                for sha, start, _ in s.file_open_on_demand_spans:
                    if sha and (sha not in first_seen or start < first_seen[sha]):
                        first_seen[sha] = start
                ordered_layers = [sha for sha, _ in sorted(first_seen.items(), key=lambda kv: kv[1])]
                layer_color = {sha: LAYER_CMAP(i % LAYER_CMAP.N) for i, sha in enumerate(ordered_layers)}

                # Per-layer prefetch micro-lanes (hatch = backslash).
                prefetch_events = sorted(s.prefetch_layer_events, key=lambda e: e[1])
                if prefetch_events:
                    n_layers = len(prefetch_events)
                    sub_h = min(bar_h, prefetch_slot_h / n_layers)
                    stack_h = n_layers * sub_h
                    slot_top = y + offsets["prefetch"] - stack_h / 2
                    for i, (sha, ev_start, ev_end) in enumerate(prefetch_events):
                        sub_y = slot_top + (i + 0.5) * sub_h
                        c = layer_color.get(sha, PREFETCH_COLOR)
                        ax.barh(sub_y, ev_end - ev_start, left=ev_start - ref,
                                height=sub_h * 0.9,
                                color=c, edgecolor="black", hatch="\\\\", linewidth=0.4)

                # Single shared file_open lane: color = layer, hatch = event type.
                for sha, ev_start, ev_end in s.file_open_cache_spans:
                    c = layer_color.get(sha, FILE_OPEN_CACHE_COLOR)
                    ax.barh(y + offsets["file_open"], ev_end - ev_start,
                            left=ev_start - ref, height=bar_h * 0.9,
                            color=c, edgecolor="black", hatch="//", linewidth=0.4)

                for sha, ev_start, ev_end in s.file_open_on_demand_spans:
                    c = layer_color.get(sha, FILE_OPEN_ON_DEMAND_COLOR)
                    ax.barh(y + offsets["file_open"], ev_end - ev_start,
                            left=ev_start - ref, height=bar_h * 0.9,
                            color=c, edgecolor="black", hatch="xx", linewidth=0.4)

                if s.bg_download_start_s is not None and s.bg_download_end_s is not None:
                    ax.barh(y + offsets["bg_download"],
                            s.bg_download_end_s - s.bg_download_start_s,
                            left=s.bg_download_start_s - ref, height=bar_h * 0.9,
                            color=BG_DOWNLOAD_COLOR, edgecolor=BG_DOWNLOAD_COLOR)

                create_left = s.create_start_s - ref
                create_w = s.create_end_s - s.create_start_s
                ax.barh(y + offsets["run"], create_w, left=create_left, height=bar_h * 0.9,
                        color=CREATE_COLOR, edgecolor=CREATE_COLOR)

                task_left = s.task_start_s - ref
                task_w = s.task_end_s - s.task_start_s
                ax.barh(y + offsets["run"], task_w, left=task_left, height=bar_h * 0.9,
                        color=RUN_COLOR, edgecolor=RUN_COLOR)

                row_max = max(
                    s.pull_end_s - ref,
                    s.task_end_s - ref,
                    (s.prefetch_end_s - ref) if s.prefetch_end_s is not None else 0.0,
                    (s.bg_download_end_s - ref) if s.bg_download_end_s is not None else 0.0,
                    max((e - ref for _, _, e in s.file_open_cache_spans), default=0.0),
                    max((e - ref for _, _, e in s.file_open_on_demand_spans), default=0.0),
                )
                x_max = max(x_max, row_max)

            ax.set_yticks([i * row_spacing for i in range(len(pcts))])
            ax.set_yticklabels([f"{p}%" for p in pcts])
            ax.set_xlabel("Time since pull start (s)")
            ax.set_title(label, fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.3, axis="x")
            ax.invert_yaxis()

        for col_idx in range(n_cols):
            axes[0][col_idx].set_xlim(0, x_max * 1.05)
        axes[0][0].set_ylabel("Partition size (%)")

        legend_handles = [
            mpatches.Patch(color=PULL_COLOR, label="pull"),
            mpatches.Patch(color=BG_DOWNLOAD_COLOR, label="bg download"),
            mpatches.Patch(color=CREATE_COLOR, label="create"),
            mpatches.Patch(color=RUN_COLOR, label="start"),
            mpatches.Patch(facecolor="lightgray", edgecolor="black",
                           hatch="\\\\", linewidth=0.4, label="prefetch (\\\\)"),
            mpatches.Patch(facecolor="lightgray", edgecolor="black",
                           hatch="//", linewidth=0.4, label="file open cache (//)"),
            mpatches.Patch(facecolor="lightgray", edgecolor="black",
                           hatch="xx", linewidth=0.4, label="file open on-demand (xx)"),
        ]
        fig.legend(
            handles=legend_handles, loc="lower right",
            bbox_to_anchor=(0.99, 0.01), ncol=len(legend_handles),
            fontsize=8, frameon=False,
        )

        fig.suptitle(
            f"Pull / Prefetch / Run timeline by stargz config ({mode}, "
            f"median run, n={N_RUNS} runs)",
            fontsize=11,
        )
        figure_footer(fig, model, base_image)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        save_figure(fig, prefetch_stages_chart_path(SCRIPT_DIR, model, base_image, mode, execution_ts))


# ── main ───────────────────────────────────────────────────────────


def main():
    log.set_verbose(VERBOSE)
    clear_artifacts(SCRIPT_DIR)
    execution_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_started = datetime.now(timezone.utc)
    log.info(f"Modes: {MODES}")
    log.info(f"Config options: {[label for _, label in CONFIG_OPTIONS]}")
    log.info(f"Partition percents: {PARTITION_PERCENTS}")
    log.info(f"Runs: {N_RUNS}")

    log.info("Pre-run cleanup...")
    for model, _ in EXPERIMENTS:
        cleanup_pull_experiment(model, SCRIPT_DIR, CFG)

    base_config_path = prefetch_stages_base_config_path(SCRIPT_DIR, execution_ts)
    os.makedirs(os.path.dirname(base_config_path), exist_ok=True)
    with open(base_config_path, "w") as f:
        f.write(read_base_config())
    log.result(f"Stargz base config snapshot saved to {base_config_path}")

    all_rows: list[PrefetchStageRow] = []
    experiments_meta: list[dict] = []
    for model, base_image in EXPERIMENTS:
        allotments, max_allowed_splits = prepare_model_splits(model, MAX_ALLOTMENTS)
        log.result(f"\n===== Experiment: {model} / {base_image} (max_splits={max_allowed_splits}) =====")
        prepare_local_registry(base_image, registry(CFG))

        results = measure(allotments, max_allowed_splits, base_image, CFG, model, base_image, execution_ts)

        rows = _build_rows(results, model, base_image)
        save_csv(rows, model, base_image, execution_ts)
        plot(results, model, base_image, execution_ts)
        all_rows.extend(rows)
        experiments_meta.append({
            "model": model,
            "base_image": base_image,
            "max_allowed_splits": max_allowed_splits,
            "partition_percents": PARTITION_PERCENTS,
        })
        cleanup_pull_experiment(model, SCRIPT_DIR, CFG)

    if all_rows:
        save_merged_csv(all_rows, execution_ts)

    write_run_json(
        prefetch_stages_run_metadata_path(SCRIPT_DIR, execution_ts),
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
