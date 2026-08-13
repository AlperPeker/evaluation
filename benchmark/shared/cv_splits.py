import json
import os
import shutil
from dataclasses import dataclass

from shared import fs, log, paths
from shared.artifacts import (
    create_base_dockerfile, create_stargz_dockerfile, write_2dfs_json,
)
from shared.config import EnvConfig
from shared.registry import plain_base_image

# Pre-split Keras CV models live at <repo>/splits/<dir>/. Each dir holds .h5
# weights + per-split .json headers and a field.json that defines the model's
# natural 2DFS field: one allotment per (row=layer-group split, col). field.json
# is our source of truth for how a model is split and in what order.
SPLITS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "splits")
)

# An OCI image caps total layers (= allotments) at ~120. One combined image holds
# all models, so their column counts must sum to <= this budget.
TOTAL_LAYER_BUDGET = 120


@dataclass(frozen=True)
class ModelSplit:
    """One CV model's ordered splits (from field.json) + its weight footprint."""
    name: str
    chunks_dir: str
    splits: list[list[str]]  # split units in layer order; each = copied file paths
    total_size: int          # bytes of .h5 weights (drives budget share)

    @property
    def n_splits(self) -> int:
        return len(self.splits)


def _copy_model(src_dir: str, dst_dir: str) -> None:
    """Copy every file except field.json into the build context (flat)."""
    fs.clear_dir(dst_dir)
    for f in sorted(os.listdir(src_dir)):
        if f == "field.json":  # read for structure, not shipped in the image
            continue
        src = os.path.join(src_dir, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(dst_dir, f))


def _load_splits(field_path: str, dst_dir: str) -> list[list[str]]:
    """Read field.json into ordered split units. Group allotments by row (layer
    order); within a row collect files in column order. Paths point at the copied
    files in dst_dir."""
    with open(field_path, encoding="utf-8") as f:
        data = json.load(f)
    by_row: dict[int, list[tuple[int, list[str]]]] = {}
    for a in data["allotments"]:
        by_row.setdefault(a["row"], []).append((a["col"], a["src"]))
    splits: list[list[str]] = []
    for row in sorted(by_row):
        files: list[str] = []
        for _, srcs in sorted(by_row[row]):
            files.extend(os.path.join(dst_dir, os.path.basename(s)) for s in srcs)
        splits.append(files)
    return splits


def prepare_cv_splits(label: str, split_dirs: list[str], work_dir: str) -> list[ModelSplit]:
    """Copy each splits/<dir> into work_dir/chunks/<label>/<dir> and read its
    field.json into ordered split units. Returns one ModelSplit per dir, in the
    given order (= stack order, bottom->top). Called once per experiment."""
    root = os.path.join(paths.chunks_dir(work_dir), label.replace("/", "--"))
    fs.clear_dir(root)
    models: list[ModelSplit] = []
    for name in split_dirs:
        src = os.path.join(SPLITS_DIR, name)
        field = os.path.join(src, "field.json")
        if not os.path.isfile(field):
            raise RuntimeError(f"No field.json in {src}")
        dst = os.path.join(root, name)
        _copy_model(src, dst)
        splits = _load_splits(field, dst)
        if not splits:
            raise RuntimeError(f"Empty field.json: {field}")
        total_size = sum(
            os.path.getsize(p)
            for s in splits for p in s if p.endswith(".h5")
        )
        models.append(ModelSplit(name, dst, splits, total_size))
    log.result(f"Prepared {len(models)} CV models into {root}")
    return models


def allocate_columns(
    models: list[ModelSplit], cap: int, budget: int = TOTAL_LAYER_BUDGET,
) -> list[int]:
    """Columns per model at a build capacity. cap is the % of the layer budget to
    use (total = round(budget*cap/100), floored at one column per model). The
    total is split across models by weight size, clamped so no model exceeds its
    own split count, water-filling the remainder."""
    n = len(models)
    caps = [m.n_splits for m in models]
    weights = [max(1, m.total_size) for m in models]
    target = max(n, round(budget * cap / 100))
    target = min(target, sum(caps))

    alloc = [1] * n  # every model gets at least one column
    while sum(alloc) < target:
        candidates = [i for i in range(n) if alloc[i] < caps[i]]
        if not candidates:
            break
        # give the next column to the most under-served model by weight
        nxt = max(candidates, key=lambda i: weights[i] / alloc[i])
        alloc[nxt] += 1
    return alloc


def full_columns(models: list[ModelSplit]) -> list[int]:
    """Column count at full capacity (used by rebuild)."""
    return allocate_columns(models, 100)


def _merge_splits(splits: list[list[str]], k: int) -> list[list[str]]:
    """Merge consecutive splits into k contiguous groups, near-even by split
    count, preserving layer order. Never reorders or size-balances."""
    k = max(1, min(k, len(splits)))
    base, extra = divmod(len(splits), k)
    groups: list[list[str]] = []
    idx = 0
    for g in range(k):
        size = base + (1 if g < extra else 0)
        groups.append([f for s in splits[idx:idx + size] for f in s])
        idx += size
    return groups


def pack_cv(
    models: list[ModelSplit], columns_per_model: list[int],
) -> tuple[list[list[str]], list[tuple[int, int]]]:
    """Merge each model's splits into its column count, then concatenate blocks in
    stack order. Returns (groups, model_ranges) where groups is the flat
    bottom->top allotment list and model_ranges[i] = (start, end) is the half-open
    group-index range owned by models[i]."""
    groups: list[list[str]] = []
    ranges: list[tuple[int, int]] = []
    for model, k in zip(models, columns_per_model):
        start = len(groups)
        groups.extend(_merge_splits(model.splits, k))
        ranges.append((start, len(groups)))
    return groups, ranges


def generate_cv_build_artifacts(
    models: list[ModelSplit], columns_per_model: list[int],
    source_image: str, cfg: EnvConfig, work_dir: str,
) -> tuple[list[list[str]], list[tuple[int, int]]]:
    """pack_cv + emit 2dfs.json / Dockerfiles for one build. Returns (groups, model_ranges)."""
    groups, ranges = pack_cv(models, columns_per_model)
    write_2dfs_json(groups, work_dir)
    create_stargz_dockerfile(groups, plain_base_image(source_image, cfg), work_dir)
    create_base_dockerfile(groups, plain_base_image(source_image, cfg), work_dir)
    return groups, ranges


def cv_packing_preview_data(
    models: list[ModelSplit], capacities: list[int],
) -> list[dict]:
    """Per-capacity packing: total columns, per-model column counts, allotment sizes (MB)."""
    out: list[dict] = []
    for cap in capacities:
        columns_per_model = allocate_columns(models, cap)
        groups, ranges = pack_cv(models, columns_per_model)
        sizes_mb = [
            round(sum(os.path.getsize(p) for p in g) / (1024 ** 2), 1) for g in groups
        ]
        out.append({
            "capacity": cap,
            "label": f"{cap}%",
            "num_layers": len(groups),
            "cols_per_model": [end - start for start, end in ranges],
            "allotment_sizes_mb": sizes_mb,
        })
    return out


def print_cv_packing_table(
    label: str, models: list[ModelSplit], capacities: list[int],
) -> None:
    """Print per-capacity (cols/model, total columns)."""
    log.result(
        f"\n=== CV packing preview: {label} ({len(models)} models, "
        f"stack: {', '.join(m.name for m in models)}) ==="
    )
    log.result(f"{'cap':>5}  {'cols/model':>24}  {'total':>5}")
    log.result("-" * 42)
    for e in cv_packing_preview_data(models, capacities):
        cpm = "[" + ", ".join(str(c) for c in e["cols_per_model"]) + "]"
        log.result(f"{e['label']:>5}  {cpm:>24}  {e['num_layers']:>5}")


def cv_split_stats(models: list[ModelSplit]) -> dict:
    """Per-model + aggregate split-pool stats (sizes in MB) for run metadata."""
    per_model = []
    total = 0
    for m in models:
        total += m.total_size
        per_model.append({
            "name": m.name,
            "n_splits": m.n_splits,
            "total_mb": round(m.total_size / (1024 ** 2), 1),
        })
    return {"total_mb": round(total / (1024 ** 2), 1), "models": per_model}
