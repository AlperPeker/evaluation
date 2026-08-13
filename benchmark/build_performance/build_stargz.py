import os
import subprocess
import time

from shared import log
from shared.build_result import BuildResult
from shared.buildctl_parser import parse_buildctl_plain
from shared.config import EnvConfig
from shared.registry import registry
from shared.services import prune_buildkit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def clear_cache() -> None:
    prune_buildkit()


def build_only(n: int, cfg: EnvConfig = None) -> BuildResult:
    target = f"{registry(cfg)}/build-perf-stargz-only:{n}"
    cmd = [
        "sudo", "buildctl", "build",
        "--progress=plain",
        "--frontend", "dockerfile.v0",
        "--opt", "filename=Dockerfile.stargz",
        "--local", f"context={SCRIPT_DIR}",
        "--local", f"dockerfile={SCRIPT_DIR}",
        "--output", f"type=image,name={target},push=false,compression=estargz,compression-level=1,oci-mediatypes=true,registry.insecure=true",
    ]

    log.info(f"=== Building with {n} split(s) (stargz) ===")
    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=SCRIPT_DIR, capture_output=True, text=True)
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        log.info(result.stderr)
        result.check_returncode()

    if log.VERBOSE:
        log.info(result.stderr)

    br = parse_buildctl_plain(result.stderr, elapsed)
    log.result(f"Build time for {n} split(s): {elapsed:.2f}s "
               f"(pull={br.pull_s:.2f} ctx={br.context_transfer_s:.2f} "
               f"build={br.build_s:.2f} export={br.export_s:.2f})")
    return br


def run_one(n: int, cfg: EnvConfig = None) -> BuildResult:
    br = build_only(n, cfg)
    clear_cache()
    return br
