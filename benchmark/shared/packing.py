import os

from shared import log


def layers_for_percent(total: int, pct: int) -> int:
    """Map a 0-100% knob onto a layer/bucket count in [1, total].

    round (not floor): for small `total` the product lands on .5
    (e.g. 3 * 50% = 1.5) and floor would collapse it onto a lower pct.
    """
    return max(1, round(total * pct / 100))


def compute_split_stats(weight_paths: list[str]) -> tuple[int, int, int]:
    """Returns (max_allotments, total_size, max_file_size).

    max_allotments = total_size // max_file_size. Floor guarantees
    target = total_size / max_allotments ≥ max_file_size, so the largest file
    (the embedding in practice) lands alone in bucket 0 and the remaining files
    best-fit into buckets of size [0.5·max_file_size, max_file_size].
    """
    if not weight_paths:
        raise RuntimeError("Empty weights pool")
    sizes = [os.path.getsize(p) for p in weight_paths]
    total_size = sum(sizes)
    max_file_size = max(sizes)
    max_allotments = max(1, total_size // max_file_size)
    log.info(
        f"splits: total={total_size/1024**2:.1f} MB  "
        f"max_file={max_file_size/1024**2:.1f} MB  "
        f"max_allotments={max_allotments}"
    )
    return max_allotments, total_size, max_file_size


def repack(
    weight_paths: list[str], num_allotments: int,
    extra_files: list[str] | None = None,
) -> list[list[str]]:
    """Best-fit pack weights + extra_files into num_allotments size-balanced buckets."""
    paths = list(weight_paths) + list(extra_files or [])
    if num_allotments <= 1:
        return [paths]

    files_with_sizes = sorted(
        ((p, os.path.getsize(p)) for p in paths),
        key=lambda x: x[1], reverse=True,
    )
    total_size = sum(sz for _, sz in files_with_sizes)
    target = total_size / num_allotments

    buckets: list[dict] = []
    for path, sz in files_with_sizes:
        best_idx = None
        best_projected = -1
        for i, b in enumerate(buckets):
            projected = b["size"] + sz
            if projected <= target and projected > best_projected:
                best_idx = i
                best_projected = projected

        if best_idx is None:
            if len(buckets) < num_allotments:
                buckets.append({"files": [path], "size": sz})
            else:
                smallest_idx = min(
                    range(len(buckets)), key=lambda i: buckets[i]["size"],
                )
                buckets[smallest_idx]["files"].append(path)
                buckets[smallest_idx]["size"] += sz
        else:
            buckets[best_idx]["files"].append(path)
            buckets[best_idx]["size"] += sz

    min_bucket_size = target * 0.5
    while len(buckets) > 1:
        smallest = min(range(len(buckets)), key=lambda i: buckets[i]["size"])
        if buckets[smallest]["size"] >= min_bucket_size:
            break
        merge_into = min(
            (i for i in range(len(buckets)) if i != smallest),
            key=lambda i: buckets[i]["size"],
        )
        buckets[merge_into]["files"].extend(buckets[smallest]["files"])
        buckets[merge_into]["size"] += buckets[smallest]["size"]
        del buckets[smallest]

    return [sorted(b["files"]) for b in buckets]
