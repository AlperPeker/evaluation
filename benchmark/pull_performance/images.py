from shared.config import EnvConfig
from shared.registry import registry, image_slug


# ── Build targets (used by prepare.py) ─────────────────────────────
# No library/ prefix — tdfs adds it automatically in the registry.


def build_name_2dfs(source_image: str, cfg: EnvConfig) -> str:
    return f"{registry(cfg)}/{image_slug(source_image)}-2dfs:latest"


def build_name_2dfs_stargz(source_image: str, cfg: EnvConfig) -> str:
    return f"{registry(cfg)}/{image_slug(source_image)}-2dfs-stargz:latest"


def build_name_2dfs_stargz_zstd(source_image: str, cfg: EnvConfig) -> str:
    return f"{registry(cfg)}/{image_slug(source_image)}-2dfs-stargz-zstd:latest"


def build_name_stargz(source_image: str, cfg: EnvConfig) -> str:
    return f"{registry(cfg)}/{image_slug(source_image)}-stargz:latest"


def build_name_base(source_image: str, cfg: EnvConfig, num_splits: int) -> str:
    return f"{registry(cfg)}/{image_slug(source_image)}-base-{num_splits}-splits:latest"


# ── Pull references (used by measure.py) ───────────────────────────
# 2dfs/2dfs-stargz need library/ prefix and allotment range tag.
# Allotment range: --0.0.0.{end_col}  pulls columns 0 through end_col.


def pull_name_2dfs(source_image: str, cfg: EnvConfig, num_allotments: int) -> str:
    end_col = num_allotments - 1
    return f"{registry(cfg)}/library/{image_slug(source_image)}-2dfs:latest--0.0.0.{end_col}"


def pull_name_2dfs_stargz(source_image: str, cfg: EnvConfig, num_allotments: int) -> str:
    end_col = num_allotments - 1
    return f"{registry(cfg)}/library/{image_slug(source_image)}-2dfs-stargz:latest--0.0.0.{end_col}"


def pull_name_2dfs_stargz_zstd(source_image: str, cfg: EnvConfig, num_allotments: int) -> str:
    end_col = num_allotments - 1
    return f"{registry(cfg)}/library/{image_slug(source_image)}-2dfs-stargz-zstd:latest--0.0.0.{end_col}"


def pull_name_stargz(source_image: str, cfg: EnvConfig) -> str:
    return build_name_stargz(source_image, cfg)


def pull_name_base(source_image: str, cfg: EnvConfig, num_splits: int) -> str:
    return build_name_base(source_image, cfg, num_splits)
