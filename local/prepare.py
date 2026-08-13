import json
import os

from huggingface_hub import hf_hub_download

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_ID = "microsoft/phi-1_5"
MODEL_FILE = "model.safetensors"
BASE_IMAGE = "ghcr.io/bohdangarchu/python:3.12-torch-esgz"


def download_model(token: str = None) -> None:
    dest = os.path.join(SCRIPT_DIR, MODEL_FILE)
    if os.path.exists(dest):
        print(f"Skipping {MODEL_FILE} (already exists)")
        return
    print(f"Downloading {MODEL_FILE}...")
    hf_hub_download(
        repo_id=MODEL_ID,
        filename=MODEL_FILE,
        token=token,
        local_dir=SCRIPT_DIR,
    )


def write_2dfs_json(output_path: str = "2dfs.json") -> None:
    data = {
        "allotments": [
            {
                "src": f"./{MODEL_FILE}",
                "dst": f"/{MODEL_FILE}",
                "row": 0,
                "col": 0,
            }
        ]
    }
    with open(os.path.join(SCRIPT_DIR, output_path), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def create_dockerfile(output_path: str = "Dockerfile.stargz") -> None:
    lines = [
        f"FROM {BASE_IMAGE}",
        f"COPY {MODEL_FILE} /{MODEL_FILE}",
    ]
    with open(os.path.join(SCRIPT_DIR, output_path), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    token = os.environ.get("HF_TOKEN")
    download_model(token=token)
    write_2dfs_json()
    create_dockerfile()
    print("Done: 2dfs.json and Dockerfile.stargz written")
