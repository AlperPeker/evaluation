import re

from shared.build_result import BuildResult


def parse_buildctl_plain(stderr: str, total_s: float) -> BuildResult:
    """Parse buildctl --progress=plain stderr output into a BuildResult.

    Matches step headers like '#N [1/3] FROM ...' and completion lines like
    '#N DONE 2.1s' or '#N CACHED'.
    """
    steps: dict[int, dict] = {}
    current_header: dict[int, str] = {}

    for line in stderr.splitlines():
        # Step header: #N <name>
        m = re.match(r"^#(\d+) (.+)$", line)
        if m:
            idx = int(m.group(1))
            name = m.group(2)
            # Only record the first header line for each step index
            if idx not in current_header:
                current_header[idx] = name
                steps[idx] = {"name": name, "duration": 0.0}

        # Done with time: #N DONE X.Xs
        m = re.match(r"^#(\d+) DONE (\d+\.?\d*)s$", line)
        if m:
            idx = int(m.group(1))
            if idx in steps:
                steps[idx]["duration"] = float(m.group(2))
            continue

        # Cached: #N CACHED
        m = re.match(r"^#(\d+) CACHED$", line)
        if m:
            idx = int(m.group(1))
            if idx in steps:
                steps[idx]["duration"] = 0.0

    pull_s = 0.0
    context_transfer_s = 0.0
    build_s = 0.0
    export_s = 0.0

    for info in steps.values():
        name = info["name"]
        dur = info["duration"]

        if "FROM " in name:
            pull_s += dur
        elif "load build context" in name:
            context_transfer_s += dur
        elif "exporting to image" in name:
            export_s += dur
        elif "COPY " in name or "RUN " in name:
            build_s += dur
        # Ignore: load build definition, load .dockerignore, load metadata

    return BuildResult(
        total_s=total_s,
        pull_s=pull_s,
        context_transfer_s=context_transfer_s,
        build_s=build_s,
        export_s=export_s,
    )
