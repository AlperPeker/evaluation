from dataclasses import dataclass


@dataclass
class BuildResult:
    total_s: float
    pull_s: float
    context_transfer_s: float
    build_s: float
    export_s: float
