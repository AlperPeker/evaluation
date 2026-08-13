import json
import os
import shutil
import struct

import numpy as np

from shared import paths

MAX_DOCKERFILE_LINE_SIZE = 65535

def snapshot_artifacts(work_dir: str, dest_dir: str) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    for src in (
        paths.tdfs_json_path(work_dir),
        paths.stargz_dockerfile_path(work_dir),
        paths.base_dockerfile_path(work_dir),
    ):
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest_dir, os.path.basename(src)))


def clear_artifacts(work_dir: str) -> None:
    for p in (
        paths.tdfs_json_path(work_dir),
        paths.stargz_dockerfile_path(work_dir),
        paths.base_dockerfile_path(work_dir),
    ):
        if os.path.exists(p):
            os.remove(p)


def mutate_chunk(path: str) -> None:
    """Flip all bits in a chunk file in-place. Calling twice restores original content."""
    with open(path, "r+b") as f:
        data = np.fromfile(f, dtype=np.uint8)
        np.bitwise_not(data, out=data)
        f.seek(0)
        data.tofile(f)


def mutate_safetensors(path: str) -> None:
    """XOR the first byte of the tensor data region with 0xFF. Calling twice
    restores original content."""
    with open(path, "r+b") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        f.seek(8 + hlen)
        b = f.read(1)
        if not b:
            return
        f.seek(8 + hlen)
        f.write(bytes([b[0] ^ 0xFF]))


def xor_first_byte(path: str) -> None:
    """XOR the first byte of the file with 0xFF. Format-agnostic (works on .h5,
    .json, etc.); calling twice restores original content."""
    with open(path, "r+b") as f:
        b = f.read(1)
        if not b:
            return
        f.seek(0)
        f.write(bytes([b[0] ^ 0xFF]))


def chunks_to_groups(chunk_paths: list[str], num_layers: int) -> list[list[str]]:
    # Partition `chunk_paths` into `num_layers` consecutive groups for split-capacity
    # benchmarks. When len(chunks) is not divisible by num_layers (75%-capacity case, e.g. 12 chunks → 9 layers), 
    # front-loaded np.array_split semantics apply: the first `len % num_layers` groups carry
    # ceil(len/num_layers) chunks each, the rest carry floor. Total bytes per build
    # are unchanged; only intra-layer chunk counts vary.
    n = len(chunk_paths)
    base, extra = divmod(n, num_layers)
    groups: list[list[str]] = []
    idx = 0
    for i in range(num_layers):
        size = base + (1 if i < extra else 0)
        groups.append(chunk_paths[idx:idx + size])
        idx += size
    return groups


def write_2dfs_json(groups: list[list[str]], work_dir: str) -> None:
    allotments = [
        {
            "src": [os.path.relpath(p, work_dir) for p in group],
            "dst": [f"/{os.path.basename(p)}" for p in group],
            "row": 0,
            "col": i,
        }
        for i, group in enumerate(groups)
    ]
    data = {"allotments": allotments}
    out_path = paths.tdfs_json_path(work_dir)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def _copy_line(group: list[str], work_dir: str) -> str:
    rels = " ".join(os.path.relpath(p, work_dir) for p in group)
    line = f"COPY {rels} /"
    if len(line) > MAX_DOCKERFILE_LINE_SIZE:
        common_dir = os.path.dirname(os.path.relpath(group[0], work_dir))
        line = f"COPY {common_dir}/ /"
    return line


def create_stargz_dockerfile(groups: list[list[str]], base_image: str, work_dir: str) -> None:
    lines = [f"FROM {base_image}"] + [_copy_line(g, work_dir) for g in groups]
    out_path = paths.stargz_dockerfile_path(work_dir)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def create_base_dockerfile(groups: list[list[str]], base_image: str, work_dir: str) -> None:
    lines = [f"FROM {base_image}"] + [_copy_line(g, work_dir) for g in groups]
    out_path = paths.base_dockerfile_path(work_dir)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
