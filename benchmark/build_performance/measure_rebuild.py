import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from shared import log
from shared.build_result import BuildResult
from build_performance.paths import (
    rebuild_csv_path, rebuild_chart_path,
    rebuild_artifacts_dir, rebuild_merged_csv_path, rebuild_run_metadata_path,
)
from shared.config import load_config
from shared.artifacts import mutate_safetensors, snapshot_artifacts, clear_artifacts
from shared.registry import prepare_local_registry, registry
from shared.services import ensure_buildkit, prune_buildkit, clear_2dfs_cache
from shared.run_metadata import write_run_json
from shared.packing import layers_for_percent
from build_performance.rebuild_common import (
    DIRECTIONS, make_methods, get_buckets_to_mutate, measure_rebuild_matrix,
    write_rebuild_csv, plot_rebuild,
)
from build_performance.prepare import (
    generate_build_artifacts, prepare_model_splits, print_packing_table,
    packing_preview_data, split_stats,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

EXPERIMENTS = [
    # ("openai-community/gpt2",        "docker.io/library/python:3.12-slim"),  # ~0.5GB     ~50 MB
    # ("Qwen/Qwen2-1.5B",              "docker.io/library/python:3.12-slim"),  # ~3.09 GB     ~3.4 GB
    # ("EleutherAI/pythia-1.4b", "docker.io/library/python:3.12-slim"),
    # ("openlm-research/open_llama_3b", "docker.io/library/python:3.12-slim"),    # ~6.0 GB     ~3.4 GB
    # ("Qwen/Qwen3.5-9B", "docker.io/ollama/ollama"),
    ("google/gemma-4-31B", "docker.io/ollama/ollama"),
]
CFG = load_config()
VERBOSE = False
MODES = ["2dfs", "2dfs-stargz", "2dfs-stargz-zstd", "stargz", "base"]
LAYERS_MUTATED_PERCENTS = [25, 50, 75, 100]
# v2:
# - n_chunks_mutated -> n_layers_mutated
# - mutation_pct -> layers_mutated_pct
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class RebuildRow:
    schema_version: int
    model: str
    base_image: str
    max_allowed_splits: int
    run: int
    layers_mutated_pct: int
    n_layers_mutated: int
    direction: str
    mode: str
    total_s: float


def mutation_preview_data(groups: list[list[str]]) -> list[dict]:
    """Structured mutated MB per (layers_mutated_pct, direction)."""
    def mutated_mb(buckets: list[list[str]]) -> float:
        return round(sum(
            os.path.getsize(p)
            for b in buckets for p in b if p.endswith(".safetensors")
        ) / (1024 ** 2), 1)

    out: list[dict] = []
    for pct in LAYERS_MUTATED_PERCENTS:
        n = layers_for_percent(len(groups), pct)
        out.append({
            "layers_mutated_pct": pct,
            "n_layers_mutated": n,
            "top_to_bottom_mb": mutated_mb(get_buckets_to_mutate(groups, n, "top_to_bottom")),
            "bottom_to_top_mb": mutated_mb(get_buckets_to_mutate(groups, n, "bottom_to_top")),
        })
    return out


def print_mutation_table(groups: list[list[str]]) -> None:
    """Print mutated MB per (layers_mutated_pct, direction)."""
    log.result(f"\n=== Mutation preview: {len(groups)} allotment(s) ===")
    log.result(f"{'pct':>5}  {'n_buckets':>9}  {'top_to_bottom (MB)':>20}  {'bottom_to_top (MB)':>20}")
    log.result("-" * 62)
    for e in mutation_preview_data(groups):
        log.result(
            f"{e['layers_mutated_pct']:>4}%  {e['n_layers_mutated']:>9}  "
            f"{e['top_to_bottom_mb']:>20.1f}  {e['bottom_to_top_mb']:>20.1f}"
        )


def measure_rebuilds(
    groups: list[list[str]], methods: list, model: str, base_image: str, max_allowed_splits: int,
) -> list[RebuildRow]:
    def select_targets(direction: str, pct: int) -> tuple[int, list[list[str]]]:
        n = layers_for_percent(len(groups), pct)
        return n, get_buckets_to_mutate(groups, n, direction)

    def make_row(run, pct, direction, mode_name, n, br: BuildResult) -> RebuildRow:
        return RebuildRow(
            schema_version=SCHEMA_VERSION, model=model, base_image=base_image,
            max_allowed_splits=max_allowed_splits, run=run, layers_mutated_pct=pct,
            n_layers_mutated=n, direction=direction, mode=mode_name, total_s=br.total_s,
        )

    return measure_rebuild_matrix(
        n_runs=CFG.rebuild_n_runs, axis_values=LAYERS_MUTATED_PERCENTS, directions=DIRECTIONS,
        methods=methods, max_allowed_splits=max_allowed_splits, select_targets=select_targets,
        mutate_ext=".safetensors", mutate_fn=mutate_safetensors, make_row=make_row,
        cooldown=CFG.build_cooldown, axis_label="mutation%",
    )


def save_csv(results: list[RebuildRow], model: str, base_image: str, execution_ts: str) -> None:
    write_rebuild_csv(rebuild_csv_path(SCRIPT_DIR, model, base_image, execution_ts), results)


def save_merged_csv(results: list[RebuildRow], execution_ts: str) -> None:
    write_rebuild_csv(rebuild_merged_csv_path(SCRIPT_DIR, execution_ts), results)


def plot(results: list[RebuildRow], model: str, base_image: str, max_allowed_splits: int, execution_ts: str) -> None:
    plot_rebuild(
        results, modes=MODES, axis_values=LAYERS_MUTATED_PERCENTS,
        axis_attr="layers_mutated_pct", xlabel="% of Layers/Allotments Updated",
        n_runs=CFG.rebuild_n_runs, model=model, base_image=base_image,
        max_allowed_splits=max_allowed_splits,
        suptitle=f"Incremental rebuild performance (mean ± std, n={CFG.rebuild_n_runs} runs)",
        output_path=rebuild_chart_path(SCRIPT_DIR, model, base_image, execution_ts),
    )


def main():
    log.set_verbose(VERBOSE)
    ensure_buildkit()
    prune_buildkit()
    clear_2dfs_cache(CFG)
    execution_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_started = datetime.now(timezone.utc)

    all_results: list[RebuildRow] = []
    experiments_meta: list[dict] = []
    for model, base_image in EXPERIMENTS:
        chunks_dir, max_allowed_splits = prepare_model_splits(model)
        log.result(f"\n===== Experiment: {model} / {base_image} (max_allowed_splits={max_allowed_splits}) =====")
        prepare_local_registry(base_image, registry(CFG))

        methods = make_methods(MODES, CFG, base_image)

        log.info(f"Preparing model at full capacity ({max_allowed_splits} buckets)...")
        groups = generate_build_artifacts(chunks_dir, max_allowed_splits, base_image, CFG)
        print_packing_table(chunks_dir, model, max_allowed_splits, ["full"], [max_allowed_splits])
        print_mutation_table(groups)

        experiments_meta.append({
            "model": model,
            "base_image": base_image,
            "max_allowed_splits": max_allowed_splits,
            "splits": split_stats(chunks_dir),
            "packing_preview": packing_preview_data(chunks_dir, ["full"], [max_allowed_splits]),
            "mutation_preview": mutation_preview_data(groups),
        })
        snapshot_artifacts(
            SCRIPT_DIR,
            rebuild_artifacts_dir(SCRIPT_DIR, execution_ts, model, base_image),
        )

        results = measure_rebuilds(groups, methods, model, base_image, max_allowed_splits)

        save_csv(results, model, base_image, execution_ts)
        plot(results, model, base_image, max_allowed_splits, execution_ts)
        all_results.extend(results)

    if all_results:
        save_merged_csv(all_results, execution_ts)

    write_run_json(
        rebuild_run_metadata_path(SCRIPT_DIR, execution_ts),
        execution_ts=execution_ts,
        started_at=run_started,
        config=asdict(CFG),
        sections={
            "modes": MODES,
            "layers_mutated_percents": LAYERS_MUTATED_PERCENTS,
            "directions": DIRECTIONS,
            "experiments": experiments_meta,
        },
    )

    clear_artifacts(SCRIPT_DIR)


if __name__ == "__main__":
    main()
