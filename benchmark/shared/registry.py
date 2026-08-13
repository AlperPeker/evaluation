import json
import os
import socket
import subprocess
import urllib.error
import urllib.request

from shared import log
from shared.config import EnvConfig


def registry(cfg: EnvConfig) -> str:
    return cfg.registry


def check_reachable(cfg: EnvConfig, timeout: float = 5.0) -> None:
    host, _, port = cfg.registry.partition(":")
    if not port:
        raise RuntimeError(f"registry {cfg.registry!r} has no port")
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            pass
    except OSError as e:
        raise RuntimeError(f"registry {cfg.registry} not reachable: {e}") from e
    log.info(f"registry {cfg.registry} reachable")


def stargz_base_image(source_image: str, cfg: EnvConfig) -> str:
    """Derive the esgz image ref in the target registry from a source image.

    ('docker.io/tensorflow/tensorflow', cfg_local)  -> 'localhost:5000/tensorflow:latest-esgz'
    ('docker.io/library/python:3.12-slim', cfg_remote) -> '131.159.25.169:5000/python:3.12-slim-esgz'
    """
    return _local_esgz_tag(source_image, cfg.registry)


def plain_base_image(source_image: str, cfg: EnvConfig) -> str:
    """Derive the plain (no conversion) image ref in the target registry from a source image."""
    return _local_plain_tag(source_image, cfg.registry)


def zstd_base_image(source_image: str, cfg: EnvConfig) -> str:
    """Derive the zstdchunked image ref in the target registry from a source image."""
    return _local_zstd_tag(source_image, cfg.registry)


def image_slug(source_image: str) -> str:
    """Extract a short slug from a source image name.

    'docker.io/library/python:3.12-slim' -> 'python-3.12-slim'
    'docker.io/tensorflow/tensorflow'     -> 'tensorflow-latest'
    """
    name = source_image.rsplit("/", 1)[-1]
    if ":" not in name:
        name += ":latest"
    return name.replace(":", "-")


def tdfs_cmd(cfg: EnvConfig, work_dir: str) -> list[str]:
    binary = cfg.tdfs_binary
    if binary.startswith("./"):
        # Local binary relative to work_dir, no sudo
        return [os.path.join(work_dir, binary[2:])]
    else:
        # Global binary: wrap with sudo + env
        cmd = ["sudo", "env"]
        if cfg.tmpdir:
            cmd.append(f"TMPDIR={cfg.tmpdir}")
        cmd.append(binary)
        if cfg.tdfs_home_dir:
            cmd += ["--home-dir", cfg.tdfs_home_dir]
        return cmd


def _local_esgz_tag(source_image: str, registry_url: str) -> str:
    """Derive local esgz tag from source image.

    'docker.io/library/python:3.12-slim' -> 'localhost:5000/python:3.12-slim-esgz'
    'docker.io/tensorflow/tensorflow'     -> 'localhost:5000/tensorflow:latest-esgz'
    """
    name_with_tag = source_image.rsplit("/", 1)[-1]  # 'python:3.12-slim' or 'tensorflow'
    if ":" not in name_with_tag:
        name_with_tag += ":latest"
    return f"{registry_url}/{name_with_tag}-esgz"


def _local_plain_tag(source_image: str, registry_url: str) -> str:
    """Derive local plain tag from source image.

    'docker.io/library/python:3.12-slim' -> 'localhost:5000/python:3.12-slim-plain'
    """
    name_with_tag = source_image.rsplit("/", 1)[-1]
    if ":" not in name_with_tag:
        name_with_tag += ":latest"
    return f"{registry_url}/{name_with_tag}-plain"


def _local_zstd_tag(source_image: str, registry_url: str) -> str:
    """Derive local zstd tag from source image.

    'docker.io/library/python:3.12-slim' -> 'localhost:5000/python:3.12-slim-zstd'
    """
    name_with_tag = source_image.rsplit("/", 1)[-1]
    if ":" not in name_with_tag:
        name_with_tag += ":latest"
    return f"{registry_url}/{name_with_tag}-zstd"


def _image_exists_in_registry(registry_url: str, name: str, tag: str) -> bool:
    """Check if image exists in registry via v2 API."""
    url = f"http://{registry_url}/v2/{name}/manifests/{tag}"
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.oci.image.manifest.v1+json, "
                  "application/vnd.docker.distribution.manifest.v2+json",
    })
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status == 200
    except (urllib.error.HTTPError, urllib.error.URLError):
        return False


def _parse_name_tag(local_tag: str, registry_url: str) -> tuple[str, str]:
    """Parse name and tag from a local registry image ref."""
    name_and_tag = local_tag.split("/", 1)[1]  # strip registry prefix
    name, tag = name_and_tag.rsplit(":", 1)
    return name, tag


def _fetch_manifest(registry_url: str, repo: str, tag: str) -> dict:
    """Fetch an image manifest from the registry v2 API.

    Resolves index manifests to the linux/amd64 entry (falls back to first entry).
    """
    base = f"http://{registry_url}/v2/{repo}/manifests"
    accept = ", ".join([
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ])
    req = urllib.request.Request(f"{base}/{tag}", headers={"Accept": accept})
    with urllib.request.urlopen(req) as resp:
        manifest = json.loads(resp.read())
    if "manifests" in manifest:
        entries = manifest["manifests"]
        chosen = None
        for entry in entries:
            plat = entry.get("platform", {})
            if plat.get("os") == "linux" and plat.get("architecture") == "amd64":
                chosen = entry
                break
        if chosen is None:
            log.info(f"no linux/amd64 in index ({len(entries)} entries); using first")
            chosen = entries[0]
        req2 = urllib.request.Request(f"{base}/{chosen['digest']}", headers={"Accept": accept})
        with urllib.request.urlopen(req2) as resp2:
            manifest = json.loads(resp2.read())
    return manifest


def fetch_layer_digests(registry_url: str, repo: str, tag: str) -> list[str]:
    """Fetch layer digests for a given image tag from the registry v2 API."""
    manifest = _fetch_manifest(registry_url, repo, tag)
    return [layer["digest"] for layer in manifest["layers"]]


def fetch_layer_sizes(registry_url: str, repo: str, tag: str) -> dict[str, int]:
    """Fetch {layer_digest: compressed_size_bytes} for an image tag."""
    manifest = _fetch_manifest(registry_url, repo, tag)
    return {layer["digest"]: int(layer["size"]) for layer in manifest["layers"]}


def save_toc(registry_url: str, repo: str, digest: str, out_path: str) -> None:
    """Run `ctr-remote fetch-toc` for a layer and write the JSON to out_path.

    On failure logs a warning and returns without raising.
    """
    ref = f"{registry_url}/{repo}@{digest}"
    log.info(f"fetch-toc {digest[:19]}... -> {os.path.basename(out_path)}")
    result = subprocess.run(
        ["sudo", "ctr-remote", "fetch-toc", ref],
        capture_output=True,
    )
    if result.returncode != 0:
        log.result(f"WARN: fetch-toc failed for {digest[:19]}... (rc={result.returncode}); skipping")
        if result.stderr:
            log.info(result.stderr.decode(errors="replace").strip())
        return
    with open(out_path, "wb") as f:
        f.write(result.stdout)


def clear_registry(cfg: EnvConfig, preserve_base: bool = False, verbose: bool = True) -> None:
    """Delete all images from the registry via v2 API.

    If preserve_base=True, skips tags ending in -plain, -esgz, or -zstd
    (the base images used as build inputs — expensive to regenerate).
    """
    reg = cfg.registry
    if verbose:
        log.info(f"Clearing registry {reg}...")
    catalog_url = f"http://{reg}/v2/_catalog"
    with urllib.request.urlopen(catalog_url) as resp:
        repos = json.loads(resp.read())["repositories"]
    for name in repos:
        tags_url = f"http://{reg}/v2/{name}/tags/list"
        with urllib.request.urlopen(tags_url) as resp:
            tags = json.loads(resp.read()).get("tags") or []
        for tag in tags:
            if preserve_base and tag.endswith(("-plain", "-esgz", "-zstd")):
                if verbose:
                    log.info(f"  Preserving base image {name}:{tag}")
                continue
            manifest_url = f"http://{reg}/v2/{name}/manifests/{tag}"
            req = urllib.request.Request(manifest_url, headers={
                "Accept": "application/vnd.docker.distribution.manifest.v2+json",
            })
            try:
                with urllib.request.urlopen(req) as resp:
                    digest = resp.headers["Docker-Content-Digest"]
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    if verbose:
                        log.info(f"  Skipping {name}:{tag} (manifest not found)")
                    continue
                raise
            delete_url = f"http://{reg}/v2/{name}/manifests/{digest}"
            del_req = urllib.request.Request(delete_url, method="DELETE")
            with urllib.request.urlopen(del_req):
                pass
            if verbose:
                log.info(f"  Deleted {name}:{tag} ({digest[:19]}...)")
    if verbose:
        log.result("Registry cleared.")


def prepare_local_registry(
    source_image: str,
    registry_url: str,
) -> str:
    """Ensure plain, esgz, and zstd base images exist in local registry.

    Pulls the plain source image, pushes it as -plain, converts to estargz (-esgz)
    and zstdchunked (-zstd), and pushes all three to the local registry. Also pushes
    library/ variants for each. Skips if all three already exist.

    Returns the local esgz image reference.
    """
    # Normalize: containerd requires the full ref with tag
    if ":" not in source_image.rsplit("/", 1)[-1]:
        source_image = f"{source_image}:latest"

    plain_tag = _local_plain_tag(source_image, registry_url)
    esgz_tag = _local_esgz_tag(source_image, registry_url)
    zstd_tag = _local_zstd_tag(source_image, registry_url)

    plain_name, plain_ver = _parse_name_tag(plain_tag, registry_url)
    esgz_name, esgz_ver = _parse_name_tag(esgz_tag, registry_url)
    zstd_name, zstd_ver = _parse_name_tag(zstd_tag, registry_url)

    log.info(f"Checking if base images exist in registry...")
    if (
        _image_exists_in_registry(registry_url, plain_name, plain_ver)
        and _image_exists_in_registry(registry_url, esgz_name, esgz_ver)
        and _image_exists_in_registry(registry_url, zstd_name, zstd_ver)
    ):
        log.info("All base images already present in registry, skipping.")
        return esgz_tag

    log.info(f"Image not found. Pulling {source_image}...")
    subprocess.run(
        ["sudo", "nerdctl", "pull", source_image],
        check=True, capture_output=True,
    )

    # ── plain ──────────────────────────────────────────────────────────
    log.info(f"Converting to OCI: {source_image} -> {plain_tag}...")
    subprocess.run(
        ["sudo", "nerdctl", "image", "convert", "--oci", source_image, plain_tag],
        check=True, capture_output=True,
    )
    log.info(f"Pushing {plain_tag} to registry...")
    subprocess.run(
        ["sudo", "nerdctl", "push", "--insecure-registry", plain_tag],
        check=True, capture_output=True,
    )
    plain_library_tag = f"{registry_url}/library/{plain_name}:{plain_ver}"
    log.info(f"Tagging and pushing library variant: {plain_library_tag}...")
    subprocess.run(
        ["sudo", "nerdctl", "tag", plain_tag, plain_library_tag],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["sudo", "nerdctl", "push", "--insecure-registry", plain_library_tag],
        check=True, capture_output=True,
    )

    # ── esgz ───────────────────────────────────────────────────────────
    log.info(f"Converting to estargz: {source_image} -> {esgz_tag}...")
    subprocess.run(
        ["sudo", "ctr-remote", "images", "convert", "--estargz", "--oci",
         "--estargz-compression-level", "1",
         source_image, esgz_tag],
        check=True, capture_output=True,
    )
    log.info(f"Pushing {esgz_tag} to registry...")
    subprocess.run(
        ["sudo", "nerdctl", "push", "--insecure-registry", esgz_tag],
        check=True, capture_output=True,
    )
    esgz_library_tag = f"{registry_url}/library/{esgz_name}:{esgz_ver}"
    log.info(f"Tagging and pushing library variant: {esgz_library_tag}...")
    subprocess.run(
        ["sudo", "nerdctl", "tag", esgz_tag, esgz_library_tag],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["sudo", "nerdctl", "push", "--insecure-registry", esgz_library_tag],
        check=True, capture_output=True,
    )

    # ── zstd ───────────────────────────────────────────────────────────
    log.info(f"Converting to zstdchunked: {source_image} -> {zstd_tag}...")
    subprocess.run(
        ["sudo", "ctr-remote", "images", "convert", "--zstdchunked", "--oci",
         "--zstdchunked-compression-level", "1",
         source_image, zstd_tag],
        check=True, capture_output=True,
    )
    log.info(f"Pushing {zstd_tag} to registry...")
    subprocess.run(
        ["sudo", "nerdctl", "push", "--insecure-registry", zstd_tag],
        check=True, capture_output=True,
    )
    zstd_library_tag = f"{registry_url}/library/{zstd_name}:{zstd_ver}"
    log.info(f"Tagging and pushing library variant: {zstd_library_tag}...")
    subprocess.run(
        ["sudo", "nerdctl", "tag", zstd_tag, zstd_library_tag],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["sudo", "nerdctl", "push", "--insecure-registry", zstd_library_tag],
        check=True, capture_output=True,
    )

    # Verify esgz (representative check)
    log.info(f"Verifying {esgz_tag} in registry...")
    if not _image_exists_in_registry(registry_url, esgz_name, esgz_ver):
        raise RuntimeError(f"Failed to push {esgz_tag} to registry")
    log.result(f"Base images ready: {plain_tag}, {esgz_tag}, {zstd_tag}")

    return esgz_tag
