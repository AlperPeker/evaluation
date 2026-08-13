import csv
import os
import subprocess
import time
from datetime import datetime, timezone

import matplotlib.pyplot as plt

from shared import log
from shared.charts import figure_footer, save_figure
from build_performance.paths import config_charts_run_dir, measure_config_csv_path, measure_config_chart_path
from shared.build_result import BuildResult
from shared.config import load_config
from shared.registry import (
    prepare_local_registry, registry, image_slug,
    plain_base_image, zstd_base_image, tdfs_cmd,
)
from shared.tdfs_parser import parse_tdfs_output
from build_performance import build_2dfs_stargz as b2s
from build_performance import build_2dfs_stargz_zstd as b2sz
from build_performance.prepare import generate_build_artifacts, prepare_model_splits, clear_chunks

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

EXPERIMENTS = [
    ("openai-community/gpt2", "docker.io/library/python:3.12-slim"),         # ~0.5GB     ~50 MB
    ("openai-community/gpt2-medium", "docker.io/tensorflow/tensorflow"),     # ~1.52 GB     ~700 MB
    ("openai-community/gpt2-large", "docker.io/ollama/ollama"),              # ~3.25 GB     ~3.4 GB
    ("openai-community/gpt2-xl", "docker.io/library/python:3.12-slim"),      # ~6.0 GB     ~50 MB
    ("openai-community/gpt2", "docker.io/tensorflow/tensorflow"),            # ~0.5GB     ~700 MB, only model size matters
]
MAX_SPLITS = 5
CFG = load_config()
VERBOSE = True
MODE = "2dfs-stargz"  # or "2dfs-stargz-zstd"
FLAG_OPTIONS: list[tuple[str, str]] = [
    ("--stargz-chunk-size 16777216", "chunk-size 16 MiB"),
    ("--stargz-chunk-size 33554432", "chunk-size 32 MiB"),
    ("--stargz-chunk-size 67108864", "chunk-size 64 MiB"),
    ("--stargz-chunk-size 134217728", "chunk-size 128 MiB"),
]

_FLAG_LABELS: dict[str, str] = {flags: label for flags, label in FLAG_OPTIONS}

ResultList = list[tuple[int, BuildResult]]


def _clear_cache() -> None:
    if MODE == "2dfs-stargz":
        b2s.clear_cache(CFG)
    elif MODE == "2dfs-stargz-zstd":
        b2sz.clear_cache(CFG)
    else:
        raise ValueError(f"Unknown mode: {MODE}")


def _build_one(n: int, flag_option: str, source_image: str) -> BuildResult:
    if MODE == "2dfs-stargz":
        base = plain_base_image(source_image, CFG)
        target = f"{registry(CFG)}/build-perf-config-stargz:{n}"
        mode_flags = ["--enable-stargz"]
    elif MODE == "2dfs-stargz-zstd":
        base = zstd_base_image(source_image, CFG)
        target = f"{registry(CFG)}/build-perf-config-stargz-zstd:{n}"
        mode_flags = ["--enable-stargz", "--use-zstd"]
    else:
        raise ValueError(f"Unknown mode: {MODE}")

    cmd = tdfs_cmd(CFG, SCRIPT_DIR) + [
        "build",
        "--platforms", "linux/amd64",
        *mode_flags,
        *flag_option.split(),
        "--force-http",
        "-f", "2dfs.json",
        base,
        target,
    ]

    log.info(f"=== Building with {n} split(s) ({MODE}, {flag_option}) ===")
    start = time.perf_counter()
    env = {**os.environ, "TMPDIR": CFG.tmpdir} if CFG.tmpdir else None
    result = subprocess.run(cmd, cwd=SCRIPT_DIR, capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        log.info(result.stdout + result.stderr)
        result.check_returncode()

    output = result.stdout + result.stderr
    if log.VERBOSE:
        log.info(output)

    br = parse_tdfs_output(output, elapsed)
    log.result(f"  {flag_option}: {elapsed:.2f}s "
               f"(pull={br.pull_s:.2f} build={br.build_s:.2f} export={br.export_s:.2f})")
    return br


def measure(model: str, max_splits: int, source_image: str) -> dict[str, ResultList]:
    results: dict[str, ResultList] = {}

    for i, (flags, label) in enumerate(FLAG_OPTIONS):
        log.info(f"\n[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] === {MODE} {label} ({flags}) ===")
        _clear_cache()
        clear_chunks(model)
        chunks_dir, _ = prepare_model_splits(model)

        flag_results: ResultList = []
        for n in range(1, max_splits + 1):
            generate_build_artifacts(chunks_dir, n, source_image, CFG)
            br = _build_one(n, flags, source_image)
            flag_results.append((n, br))

        results[flags] = flag_results

        if i < len(FLAG_OPTIONS) - 1:
            log.info(f"\nSleeping {CFG.build_cooldown}s before next flag option...")
            time.sleep(CFG.build_cooldown)

    return results


def save_csv(
    splits: list[int],
    results: dict[str, list[BuildResult]],
    model: str,
    base_image: str,
    execution_ts: str,
) -> None:
    output_path = measure_config_csv_path(SCRIPT_DIR, model, base_image, MODE, len(splits), execution_ts)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    header = ["splits"]
    for flags, label in FLAG_OPTIONS:
        slug = label.replace(" ", "_")
        header.extend([f"{slug}_total_s", f"{slug}_pull_s", f"{slug}_ctx_s", f"{slug}_build_s", f"{slug}_export_s"])

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, n in enumerate(splits):
            row: list = [n]
            for flags, label in FLAG_OPTIONS:
                br = results[flags][i]
                row.extend([f"{br.total_s:.4f}", f"{br.pull_s:.4f}", f"{br.context_transfer_s:.4f}",
                            f"{br.build_s:.4f}", f"{br.export_s:.4f}"])
            writer.writerow(row)
    log.result(f"Results saved to {output_path}")


def plot(
    results: dict[str, ResultList],
    model: str,
    base_image: str,
    execution_ts: str,
) -> None:
    splits = [n for n, _ in next(iter(results.values()))]
    os.makedirs(config_charts_run_dir(SCRIPT_DIR, execution_ts), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for flags, flag_results in results.items():
        ax.plot(splits, [br.total_s for _, br in flag_results], marker="o", label=_FLAG_LABELS[flags])

    ax.set_xlabel("Number of splits")
    ax.set_ylabel("Build time (s)")
    ax.set_title(f"Build performance by config ({MODE})")
    ax.set_xticks(splits)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    figure_footer(fig, model, base_image)
    fig.tight_layout()

    output_path = measure_config_chart_path(SCRIPT_DIR, model, base_image, MODE, len(splits), execution_ts)
    save_figure(fig, output_path)


def main():
    log.set_verbose(VERBOSE)
    execution_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log.info(f"Mode: {MODE}")
    log.info(f"Flag options: {[f'{label} ({flags})' for flags, label in FLAG_OPTIONS]}")

    for model, base_image in EXPERIMENTS:
        log.result(f"\n===== Experiment: {model} / {base_image} =====")
        prepare_local_registry(base_image, registry(CFG))

        results = measure(model, MAX_SPLITS, base_image)

        splits = [n for n, _ in next(iter(results.values()))]
        brs = {flags: [br for _, br in fo_results] for flags, fo_results in results.items()}

        log.result(f"\n=== Results ({MODE}) ===")
        col = max(len(label) for _, label in FLAG_OPTIONS) + 2
        header_flags = "  ".join(f"{label:>{col}}" for _, label in FLAG_OPTIONS)
        log.result(f"{'splits':>8}  {header_flags}")
        log.result("-" * (10 + (col + 2) * len(FLAG_OPTIONS)))
        for i, n in enumerate(splits):
            row = "  ".join(f"{brs[flags][i].total_s:>{col}.2f}" for flags, _ in FLAG_OPTIONS)
            log.result(f"{n:>8}  {row}")

        save_csv(splits, brs, model, base_image, execution_ts)
        plot(results, model, base_image, execution_ts)


if __name__ == "__main__":
    main()
