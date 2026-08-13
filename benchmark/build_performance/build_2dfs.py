import os
import subprocess
import time

from shared import log
from shared.build_result import BuildResult
from shared.config import EnvConfig
from shared.registry import plain_base_image, tdfs_cmd, registry
from shared.services import clear_2dfs_cache
from shared.tdfs_parser import parse_tdfs_output

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def clear_cache(cfg: EnvConfig) -> None:
    clear_2dfs_cache(cfg)


def build_only(n: int, cfg: EnvConfig, source_image: str = "") -> BuildResult:
    target = f"{registry(cfg)}/build-perf:{n}"
    cmd = tdfs_cmd(cfg, SCRIPT_DIR) + [
        "build",
        "--platforms", "linux/amd64",
        "--force-http",
        "--compression-level", "1",
        "-f", "2dfs.json",
        plain_base_image(source_image, cfg),
        target,
    ]

    log.info(f"=== Building with {n} split(s) (2dfs) ===")
    start = time.perf_counter()
    env = {**os.environ, "TMPDIR": cfg.tmpdir} if cfg.tmpdir else None
    result = subprocess.run(cmd, cwd=SCRIPT_DIR, capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        log.info(result.stdout + result.stderr)
        result.check_returncode()

    output = result.stdout + result.stderr
    if log.VERBOSE:
        log.info(output)

    br = parse_tdfs_output(output, elapsed)
    log.result(f"Build time for {n} split(s): {elapsed:.2f}s "
               f"(pull={br.pull_s:.2f} build={br.build_s:.2f} export={br.export_s:.2f})")
    return br


def run_one(n: int, cfg: EnvConfig, source_image: str = "") -> BuildResult:
    br = build_only(n, cfg, source_image)
    clear_cache(cfg)
    return br
