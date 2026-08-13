import os
from dataclasses import dataclass
from typing import Optional
import yaml


@dataclass
class EnvConfig:
    registry: str
    tdfs_home_dir: Optional[str]
    tmpdir: Optional[str]
    data_volume: Optional[str]
    hf_home: Optional[str]
    tdfs_binary: str
    build_cooldown: int
    pull_cooldown: int
    build_n_runs: int
    build_with_resource: bool
    rebuild_n_runs: int
    pull_n_runs: int
    refresh_n_runs: int


_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def load_config() -> EnvConfig:
    with open(_CONFIG_PATH) as f:
        data = yaml.safe_load(f)
    return EnvConfig(**data)


def build_tmpdir(cfg: EnvConfig) -> str:
    """TMPDIR builds stage through: config value, else $TMPDIR, else Go's /tmp default."""
    return cfg.tmpdir or os.environ.get("TMPDIR") or "/tmp"


def data_volume(cfg: EnvConfig) -> str:
    """Filesystem the workload writes to; disk_usage is reported for its mount. Default root."""
    return cfg.data_volume or "/"
