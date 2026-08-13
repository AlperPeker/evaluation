import os
from dataclasses import dataclass
from datetime import datetime, timezone

from shared import log
from shared.build_result import BuildResult
from build_performance.paths import (
    rebuild_chart_path, rebuild_artifacts_dir,
    rebuild_multimodel_merged_csv_path, rebuild_run_metadata_path,
)
from shared.config import load_config
from shared.artifacts import mutate_chunk, snapshot_artifacts, clear_artifacts
from shared.registry import prepare_local_registry, registry
from shared.services import ensure_buildkit, prune_buildkit, clear_2dfs_cache
from shared.run_metadata import write_run_json
from shared import cv_splits as cv
from build_performance.measure import MultiModel
from build_performance.rebuild_common import (
    DIRECTIONS, make_methods, get_buckets_to_mutate, measure_rebuild_matrix,
    write_rebuild_csv, plot_rebuild,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

EXPERIMENTS = [
    MultiModel("cv-4model",
               ["resnet50_seperated", "deeplab_v3_seperated",
                "efficientnet_v2M_seperated", "yolov3_seperated"],
               "docker.io/tensorflow/tensorflow"),
]
CFG = load_config()
VERBOSE = False
MODES = ["2dfs", "2dfs-stargz", "2dfs-stargz-zstd", "stargz", "base"]
# how many whole models are updated (1..N). Replaces the single-model
# layers_mutated_pct axis. Each model is one allotment. direction picks the
# top-k (last in EXPERIMENTS order) or bottom-k (first in EXPERIMENTS order).
MODELS_UPDATED = [1, 2, 3, 4]
SCHEMA_VERSION = 4


@dataclass(frozen=True)
class CvRebuildRow:
    schema_version: int
    model: str           # MultiModel.label
    base_image: str
    max_allowed_splits: int
    run: int
    models_updated: int
    n_allotments_mutated: int  # columns actually touched (differs per direction)
    direction: str
    models_mutated: str  # ";"-joined model names, stack order (bottom->top)
    mode: str
    total_s: float


def cv_buckets_for(
    groups: list[list[str]], ranges: list[tuple[int, int]], k: int, direction: str,
) -> list[list[str]]:
    """Allotment buckets of the k model-blocks selected from the top or bottom
    of the stack."""
    chosen = get_buckets_to_mutate(list(range(len(ranges))), k, direction)
    return [g for idx in chosen for g in groups[ranges[idx][0]:ranges[idx][1]]]


def cv_models_for(
    model_names: list[str], n_models: int, k: int, direction: str,
) -> list[str]:
    """The k model names selected from the top or bottom of the stack, in stack
    order (bottom->top)."""
    chosen = get_buckets_to_mutate(list(range(n_models)), k, direction)
    return [model_names[i] for i in chosen]


def cv_mutation_preview_data(
    groups: list[list[str]], ranges: list[tuple[int, int]],
) -> list[dict]:
    """Per (#models, direction): columns + mutated MB."""
    def mutated_mb(buckets: list[list[str]]) -> float:
        return round(sum(
            os.path.getsize(p) for b in buckets for p in b if p.endswith(".h5")
        ) / (1024 ** 2), 1)

    out: list[dict] = []
    for k in MODELS_UPDATED:
        ttb = cv_buckets_for(groups, ranges, k, "top_to_bottom")
        btt = cv_buckets_for(groups, ranges, k, "bottom_to_top")
        out.append({
            "models_updated": k,
            "top_to_bottom_cols": len(ttb), "top_to_bottom_mb": mutated_mb(ttb),
            "bottom_to_top_cols": len(btt), "bottom_to_top_mb": mutated_mb(btt),
        })
    return out


def print_cv_mutation_table(groups: list[list[str]], ranges: list[tuple[int, int]]) -> None:
    log.result(f"\n=== CV mutation preview: {len(ranges)} models, {len(groups)} allotment(s) ===")
    log.result(f"{'#models':>7}  {'top_to_bottom (cols/MB)':>24}  {'bottom_to_top (cols/MB)':>24}")
    log.result("-" * 62)
    for e in cv_mutation_preview_data(groups, ranges):
        ttb = f"{e['top_to_bottom_cols']}/{e['top_to_bottom_mb']:.1f}"
        btt = f"{e['bottom_to_top_cols']}/{e['bottom_to_top_mb']:.1f}"
        log.result(f"{e['models_updated']:>7}  {ttb:>24}  {btt:>24}")


def measure_cv_rebuilds(
    groups: list[list[str]], ranges: list[tuple[int, int]], methods: list,
    model_names: list[str], model: str, base_image: str, max_allowed_splits: int,
) -> list[CvRebuildRow]:
    def select_targets(direction: str, k: int) -> tuple[int, list[list[str]]]:
        buckets = cv_buckets_for(groups, ranges, k, direction)
        return len(buckets), buckets

    def make_row(run, k, direction, mode_name, n_allotments, br: BuildResult) -> CvRebuildRow:
        return CvRebuildRow(
            schema_version=SCHEMA_VERSION, model=model, base_image=base_image,
            max_allowed_splits=max_allowed_splits, run=run, models_updated=k,
            n_allotments_mutated=n_allotments, direction=direction,
            models_mutated=";".join(cv_models_for(model_names, len(ranges), k, direction)),
            mode=mode_name, total_s=br.total_s,
        )

    return measure_rebuild_matrix(
        n_runs=CFG.rebuild_n_runs, axis_values=MODELS_UPDATED, directions=DIRECTIONS,
        methods=methods, max_allowed_splits=max_allowed_splits, select_targets=select_targets,
        mutate_ext=".h5", mutate_fn=mutate_chunk, make_row=make_row,
        cooldown=CFG.build_cooldown, axis_label="#models",
    )


def plot(results: list[CvRebuildRow], model: str, base_image: str, max_allowed_splits: int, execution_ts: str) -> None:
    plot_rebuild(
        results, modes=MODES, axis_values=MODELS_UPDATED,
        axis_attr="models_updated", xlabel="# Models Updated",
        n_runs=CFG.rebuild_n_runs, model=model, base_image=base_image,
        max_allowed_splits=max_allowed_splits,
        suptitle=f"Multi-model incremental rebuild (mean ± std, n={CFG.rebuild_n_runs} runs)",
        output_path=rebuild_chart_path(SCRIPT_DIR, model, base_image, execution_ts),
    )


def main():
    log.set_verbose(VERBOSE)
    ensure_buildkit()
    prune_buildkit()
    clear_2dfs_cache(CFG)
    execution_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_started = datetime.now(timezone.utc)

    all_results: list[CvRebuildRow] = []
    experiments_meta: list[dict] = []
    for exp in EXPERIMENTS:
        models = cv.prepare_cv_splits(exp.label, exp.split_dirs, SCRIPT_DIR)
        max_allowed_splits = sum(cv.full_columns(models))
        log.result(f"\n===== CV rebuild: {exp.label} / {exp.base_image} (max_allowed_splits={max_allowed_splits}) =====")
        prepare_local_registry(exp.base_image, registry(CFG))

        methods = make_methods(MODES, CFG, exp.base_image)

        log.info(f"Preparing CV image at full split ({max_allowed_splits} allotments)...")
        groups, ranges = cv.generate_cv_build_artifacts(
            models, cv.full_columns(models), exp.base_image, CFG, SCRIPT_DIR,
        )
        cv.print_cv_packing_table(exp.label, models, [100])
        print_cv_mutation_table(groups, ranges)

        experiments_meta.append({
            "model": exp.label,
            "base_image": exp.base_image,
            "max_allowed_splits": max_allowed_splits,
            "splits": cv.cv_split_stats(models),
            "mutation_preview": cv_mutation_preview_data(groups, ranges),
        })
        snapshot_artifacts(
            SCRIPT_DIR,
            rebuild_artifacts_dir(SCRIPT_DIR, execution_ts, exp.label, exp.base_image),
        )

        results = measure_cv_rebuilds(
            groups, ranges, methods, [m.name for m in models],
            exp.label, exp.base_image, max_allowed_splits,
        )

        plot(results, exp.label, exp.base_image, max_allowed_splits, execution_ts)
        all_results.extend(results)

    if all_results:
        write_rebuild_csv(rebuild_multimodel_merged_csv_path(SCRIPT_DIR, execution_ts), all_results)

    write_run_json(
        rebuild_run_metadata_path(SCRIPT_DIR, execution_ts),
        execution_ts=execution_ts,
        started_at=run_started,
        config={
            "registry": registry(CFG),
            "tdfs_binary": CFG.tdfs_binary,
            "rebuild_n_runs": CFG.rebuild_n_runs,
            "build_cooldown": CFG.build_cooldown,
        },
        sections={
            "modes": MODES,
            "models_updated": MODELS_UPDATED,
            "directions": DIRECTIONS,
            "experiments": experiments_meta,
        },
    )

    clear_artifacts(SCRIPT_DIR)


if __name__ == "__main__":
    main()
