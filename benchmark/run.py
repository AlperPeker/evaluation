import os
import socket
import time

from shared import log, notify, paths, registry
from shared.config import load_config
from shared.model import cleanup_build_artifacts, cleanup_pull_artifacts

import build_performance.measure as bm
import build_performance.measure_rebuild as bmr
import build_performance.measure_cv_rebuild as bcr
import pull_performance.measure as pm
import pull_performance.measure_refresh as pmr
import pull_performance.measure_prefetch_stages as pps
import pull_performance.measure_stargz_config as psc

_BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))


def run_phases():
    cfg = load_config()
    registry.check_reachable(cfg)

    log.result("=== Phase 1: Build Performance ===")
    bm.main()

    log.result("=== Phase 2: Build Rebuild Performance ===")
    bmr.main()

    # optional - multi-model rebuild
    # log.result("=== Phase 2b: CV Multi-Model Rebuild Performance ===")
    # bcr.main()

    log.result("=== clear build artifacts ===")
    cleanup_build_artifacts(paths.build_perf_dir(_BENCHMARK_DIR), cfg)

    log.result("=== Phase 3: Pull Performance ===")
    pm.main()

    log.result("=== Phase 4: Refresh Performance ===")
    pmr.main()

    # optional
    # log.result("=== Phase 5: Prefetch Stages ===")
    # pps.main()

    # log.result("=== Phase 6: Stargz Config ===")
    # psc.main()


def main():
    log.set_verbose(True)
    host = socket.gethostname()
    start = time.time()
    try:
        run_phases()
    except BaseException as e:
        mins = (time.time() - start) / 60
        notify.notify(
            f"run.py FAILED on {host} after {mins:.1f} min\n{type(e).__name__}: {e}",
            title="Benchmark failed",
            tags="x",
            priority="high",
        )
        raise

    mins = (time.time() - start) / 60
    notify.notify(
        f"run.py finished OK on {host} in {mins:.1f} min",
        title="Benchmark done",
        tags="white_check_mark",
    )


if __name__ == "__main__":
    main()
