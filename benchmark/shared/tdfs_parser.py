import re

from shared.build_result import BuildResult


def parse_tdfs_output(output: str, total_s: float) -> BuildResult:
    """Parse tdfs build stderr/stdout output into a BuildResult.

    Looks for:
      - 'Image index retrieved (total download took X.XXms)' or '(total download took X.XXs)'
      - 'Build completed ⚒️ (X.XXs)'
      - 'Done! ✅ (X.XXs)'
    """
    pull_s = 0.0
    build_s = 0.0
    done_s = 0.0

    for line in output.splitlines():
        # Image index retrieved (total download took 25.755793ms)
        m = re.search(r"Image index retrieved \(total download took ([\d.]+)(ms|s)\)", line)
        if m:
            val = float(m.group(1))
            unit = m.group(2)
            pull_s = val / 1000.0 if unit == "ms" else val
            continue

        # Build completed ⚒️ (0.117000s)
        m = re.search(r"Build completed\s+⚒️?\s*\(([\d.]+)s\)", line)
        if m:
            build_s = float(m.group(1))
            continue

        # Done! ✅ (0.143000s)
        m = re.search(r"Done!\s+✅?\s*\(([\d.]+)s\)", line)
        if m:
            done_s = float(m.group(1))

    # tdfs has no context transfer (reads files directly)
    # export_s is the push time: total minus the internal done time
    # If done_s was not parsed, fall back to total - build - pull
    export_s = total_s - done_s if done_s > 0 else total_s - build_s - pull_s

    return BuildResult(
        total_s=total_s,
        pull_s=pull_s,
        context_transfer_s=0.0,
        build_s=build_s,
        export_s=export_s,
    )
