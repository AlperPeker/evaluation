import csv
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import matplotlib.pyplot as plt

from shared import log
from shared.charts import figure_footer, save_figure
from build_performance.paths import compression_charts_dir, compression_csv_path, compression_chart_path
from shared.config import load_config
from shared.registry import prepare_local_registry, registry, plain_base_image, tdfs_cmd, image_slug
from shared.build_result import BuildResult
from shared.services import clear_2dfs_cache
from shared.tdfs_parser import parse_tdfs_output
from build_performance.prepare import generate_build_artifacts, prepare_model_splits, clear_chunks

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL = "openai-community/gpt2-medium"
SOURCE_IMAGE = "docker.io/library/python:3.12-slim"
MAX_SPLITS = 2
CFG = load_config()

LEVELS = [
    ("level_n1", "level -1 (default)", "-1"),
    ("level_1",  "level 1 (best-speed)", "1"),
]


@dataclass
class RunResult:
    build: BuildResult
    image_size: str  # e.g. "1.4 GB"


def clear_cache() -> None:
    clear_2dfs_cache(CFG)


def build_one(n: int, level: str) -> BuildResult:
    target = f"{registry(CFG)}/compression-test-level{level.replace('-', 'n')}:{n}"
    cmd = tdfs_cmd(CFG, SCRIPT_DIR) + [
        "build",
        "--platforms", "linux/amd64",
        "--force-http",
        "-f", "2dfs.json",
        "--compression-level", level,
        plain_base_image(SOURCE_IMAGE, CFG),
        target,
    ]

    env = {**os.environ, "TMPDIR": CFG.tmpdir} if CFG.tmpdir else None
    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=SCRIPT_DIR, capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        log.info(result.stdout + result.stderr)
        result.check_returncode()

    output = result.stdout + result.stderr
    if log.VERBOSE:
        log.info(output)

    br = parse_tdfs_output(output, elapsed)
    log.result(f"  splits={n} total={elapsed:.2f}s pull={br.pull_s:.2f}s build={br.build_s:.2f}s export={br.export_s:.2f}s")
    return br


def get_image_size() -> str:
    """Run tdfs image ls and parse the Size column from the OCI+2DFS row."""
    cmd = tdfs_cmd(CFG, SCRIPT_DIR) + ["image", "ls"]
    result = subprocess.run(cmd, cwd=SCRIPT_DIR, capture_output=True, text=True)
    log.info(result.stdout)
    for line in result.stdout.splitlines():
        if "OCI+2DFS" not in line:
            continue
        # Line looks like:
        #  1  10.10.1.2:5000/...  tag  OCI+2DFS  1.3 GB  <hash>
        # Size is the token matching \d+\.\d+ [KMGTPE]?B before the 64-char hex hash
        m = re.search(r"([\d.]+\s+[KMGTPE]?B)\s+[0-9a-f]{64}", line)
        if m:
            return m.group(1).strip()
    return "unknown"


def run_level(key: str, label: str, level: str) -> list[tuple[int, RunResult]]:
    log.result(f"\n[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] === {label} ===")
    clear_cache()
    results = []
    for n in range(1, MAX_SPLITS + 1):
        log.info(f"\n=== Preparing {n} split(s) ===")
        clear_chunks(MODEL)
        chunks_dir, _ = prepare_model_splits(MODEL)
        generate_build_artifacts(chunks_dir, n, SOURCE_IMAGE, CFG)
        br = build_one(n, level)
        size = get_image_size()
        log.result(f"  image size: {size}")
        clear_cache()
        results.append((n, RunResult(build=br, image_size=size)))
    return results


def save_csv(all_results: list[tuple[str, str, list[tuple[int, RunResult]]]]) -> None:
    path = compression_csv_path(SCRIPT_DIR, MODEL, SOURCE_IMAGE, MAX_SPLITS)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    header = ["splits"]
    for key, _, _ in all_results:
        header.extend([f"{key}_total_s", f"{key}_pull_s", f"{key}_ctx_s",
                       f"{key}_build_s", f"{key}_export_s", f"{key}_size"])

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(MAX_SPLITS):
            n = i + 1
            row: list = [n]
            for _, _, results in all_results:
                _, rr = results[i]
                br = rr.build
                row.extend([f"{br.total_s:.4f}", f"{br.pull_s:.4f}", f"{br.context_transfer_s:.4f}",
                             f"{br.build_s:.4f}", f"{br.export_s:.4f}", rr.image_size])
            writer.writerow(row)
    log.result(f"Results saved to {path}")


def plot(all_results: list[tuple[str, str, list[tuple[int, RunResult]]]]) -> None:
    os.makedirs(compression_charts_dir(SCRIPT_DIR), exist_ok=True)
    splits = [n for n, _ in all_results[0][2]]

    def to_mb(size_str: str) -> float:
        m = re.match(r"([\d.]+)\s*([KMGTPE]?)B", size_str)
        if not m:
            return 0.0
        val, unit = float(m.group(1)), m.group(2)
        multipliers = {"": 1/1024/1024, "K": 1/1024, "M": 1.0, "G": 1024.0, "T": 1024.0**2}
        return val * multipliers.get(unit, 1.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for _, label, results in all_results:
        ax1.plot(splits, [rr.build.total_s - rr.build.pull_s for _, rr in results], marker="o", label=label)
    ax1.set_xlabel("Number of splits")
    ax1.set_ylabel("Build time (s)")
    ax1.set_title("Build time by compression level (excl. pull)")
    ax1.set_xticks(splits)
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    for _, label, results in all_results:
        ax2.plot(splits, [to_mb(rr.image_size) for _, rr in results], marker="o", label=label)
    ax2.set_xlabel("Number of splits")
    ax2.set_ylabel("Image size (MB)")
    ax2.set_title("Image size by compression level")
    ax2.set_xticks(splits)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    figure_footer(fig, MODEL, SOURCE_IMAGE)
    fig.tight_layout()
    save_figure(fig, compression_chart_path(SCRIPT_DIR, MODEL, SOURCE_IMAGE, MAX_SPLITS))


def main():
    log.set_verbose(True)

    prepare_local_registry(SOURCE_IMAGE, registry(CFG))

    all_results: list[tuple[str, str, list[tuple[int, RunResult]]]] = []
    for key, label, level in LEVELS:
        results = run_level(key, label, level)
        all_results.append((key, label, results))

    log.result("\n=== Summary ===")
    col = 20
    header = f"{'splits':>8}"
    for _, label, _ in all_results:
        header += f"  {(label + ' time'):>{col}}  {(label + ' size'):>{col}}"
    log.result(header)
    log.result("-" * (8 + (col * 2 + 6) * len(all_results)))
    for i in range(MAX_SPLITS):
        n = i + 1
        row = f"{n:>8}"
        for _, _, results in all_results:
            _, rr = results[i]
            row += f"  {rr.build.total_s:>{col}.2f}  {rr.image_size:>{col}}"
        log.result(row)

    save_csv(all_results)
    plot(all_results)


if __name__ == "__main__":
    main()
