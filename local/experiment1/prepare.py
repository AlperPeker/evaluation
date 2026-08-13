import json
import os
import shutil

from huggingface_hub import hf_hub_download, list_repo_files

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_IMAGE = "ghcr.io/bohdangarchu/python:3.10-esgz"

# Each model must have exactly one top-level .safetensors file under 700MB
MODELS = [
    {"repo_id": "openai-community/gpt2-medium", "local_name": "gpt2-medium.safetensors"},
    {"repo_id": "microsoft/phi-1", "local_name": "phi-1.safetensors"},
    {"repo_id": "FacebookAI/roberta-large", "local_name": "roberta-large.safetensors"},
]


def find_safetensors_file(repo_id: str, token: str = None) -> str:
    files = [f for f in list_repo_files(repo_id, token=token) if f.endswith(".safetensors") and "/" not in f]
    if len(files) != 1:
        raise ValueError(f"Expected exactly 1 top-level .safetensors file in {repo_id}, found: {files}")
    return files[0]


def download_models(token: str = None) -> list[str]:
    local_names = []
    for m in MODELS:
        dest = os.path.join(SCRIPT_DIR, m["local_name"])
        if os.path.exists(dest):
            print(f"Skipping {m['local_name']} (already exists)")
        else:
            hf_name = find_safetensors_file(m["repo_id"], token=token)
            tmp_dir = os.path.join(SCRIPT_DIR, ".cache", m["repo_id"].replace("/", "_"))
            os.makedirs(tmp_dir, exist_ok=True)
            print(f"Downloading {m['repo_id']}/{hf_name}...")
            hf_hub_download(
                repo_id=m["repo_id"],
                filename=hf_name,
                token=token,
                local_dir=tmp_dir,
            )
            shutil.move(os.path.join(tmp_dir, hf_name), dest)
        local_names.append(m["local_name"])
    return local_names


def write_2dfs_json(local_names: list[str], output_path: str = "2dfs.json") -> None:
    data = {
        "allotments": [
            {
                "src": f"./{name}",
                "dst": f"/{name}",
                "row": 0,
                "col": idx,
            }
            for idx, name in enumerate(local_names)
        ]
    }
    with open(os.path.join(SCRIPT_DIR, output_path), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def create_full_dockerfile(local_names: list[str], output_path: str = "Dockerfile.stargz") -> None:
    lines = [f"FROM {BASE_IMAGE}"]
    for name in local_names:
        lines.append(f"COPY {name} /{name}")
    with open(os.path.join(SCRIPT_DIR, output_path), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def create_base_dockerfile(local_names: list[str], col: int, output_path: str) -> None:
    lines = [f"FROM {BASE_IMAGE}"]
    lines.append(f"COPY {local_names[col]} /{local_names[col]}")
    with open(os.path.join(SCRIPT_DIR, output_path), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    token = os.environ.get("HF_TOKEN")
    local_names = download_models(token=token)
    write_2dfs_json(local_names)
    create_full_dockerfile(local_names)
    for i in range(len(local_names)):
        create_base_dockerfile(local_names, i, f"Dockerfile.base.{i + 1}")
    print("Done: 2dfs.json, Dockerfile.stargz, Dockerfile.base.1/2/3 written")
