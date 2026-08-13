import json
import os
import re
import subprocess
import time

from shared import log

JOURNAL_KV_REGEX = re.compile(r'(\w+)=("(?:[^"\\]|\\.)*"|\S+)')
STARGZ_ROOT = "/var/lib/containerd-stargz-grpc"
OVERLAYFS_ROOT = "/var/lib/containerd/io.containerd.snapshotter.v1.overlayfs"


def parse_journal_kv(text: str) -> dict[str, str]:
    """Parse a logrus-formatted log line into a key→value dict."""
    result = {}
    for m in JOURNAL_KV_REGEX.finditer(text):
        result[m.group(1)] = m.group(2).strip('"')
    return result


def collect_stargz_journal_since(since_s: float) -> list[dict]:
    """Return all stargz-snapshotter journal entries since a Unix timestamp."""
    result = subprocess.run(
        ["sudo", "journalctl", "-u", "stargz-snapshotter",
         f"--since=@{since_s:.0f}", "--output", "json", "--no-pager"],
        capture_output=True, text=True, check=True,
    )
    entries = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return entries


def save_stargz_run_log(pull_start_s: float, run_end_s: float, log_path: str) -> None:
    entries = collect_stargz_journal_since(pull_start_s)
    run_end_us = run_end_s * 1_000_000
    windowed = [e for e in entries if int(e.get("__REALTIME_TIMESTAMP", 0)) <= run_end_us]
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(windowed, f, indent=2)
    log.info(f"  Saved {len(windowed)} log entries → {log_path}")


def clear_2dfs_cache(cfg) -> None:
    log.info("Clearing 2dfs cache...")
    # tdfs is invoked via sudo so it runs as root and defaults to /root/.2dfs;
    # os.path.expanduser resolves ~ for the current (non-root) user and would
    # clear the wrong directory, leaving the actual cache untouched.
    home = cfg.tdfs_home_dir or "/root/.2dfs"
    for sub in ("blobs", "uncompressed-keys", "index"):
        subprocess.run(["sudo", "rm", "-rf", f"{home}/{sub}"], check=True)


def clear_stargz_cache() -> None:
    """Full stargz wipe: unmount FUSE, rm -rf stargz root, restart containerd."""
    log.info("Clearing stargz cache...")
    subprocess.run("sudo systemctl stop stargz-snapshotter", shell=True, check=True)
    subprocess.run(
        f"grep '{STARGZ_ROOT}/snapshotter/snapshots' /proc/mounts"
        f" | awk '{{print $2}}' | xargs -r sudo umount -l",
        shell=True, check=True,
    )
    subprocess.run(f"sudo bash -c 'rm -rf {STARGZ_ROOT}/*'", shell=True, check=True)
    subprocess.run("sudo nerdctl image rm -f $(sudo nerdctl images -q) 2>/dev/null || true", shell=True)
    subprocess.run("sudo ctr content rm $(sudo ctr content ls -q) 2>/dev/null || true", shell=True)
    subprocess.run(
        "sudo rm -f /var/lib/containerd/io.containerd.metadata.v1.bolt/meta.db",
        shell=True, check=True,
    )
    subprocess.run("sudo systemctl start stargz-snapshotter", shell=True, check=True)
    subprocess.run("sudo systemctl restart containerd", shell=True, check=True)


def clear_overlayfs_cache() -> None:
    """Wipe the overlayfs snapshotter state. containerd must be stopped and the
    overlay mounts unmounted first, otherwise live mounts get partially removed
    and later pulls hit 'file exists' on snapshot rename. Pair with
    clear_stargz_cache (which drops meta.db) before an overlayfs pull so the two
    snapshotter states stay consistent."""
    log.info("Clearing overlayfs cache...")
    subprocess.run("sudo systemctl stop containerd", shell=True, check=True)
    subprocess.run(
        f"grep '{OVERLAYFS_ROOT}/snapshots' /proc/mounts"
        " | awk '{print $2}' | sort -r | xargs -r sudo umount -l",
        shell=True, check=True,
    )
    subprocess.run(
        f"sudo bash -c 'rm -rf {OVERLAYFS_ROOT}/metadata.db {OVERLAYFS_ROOT}/snapshots/*'",
        shell=True, check=True,
    )
    subprocess.run("sudo systemctl start containerd", shell=True, check=True)


def prune_buildkit() -> None:
    ensure_buildkit()
    log.info("Pruning buildkit cache...")
    subprocess.run(["sudo", "buildctl", "prune", "--all"], check=True, capture_output=not log.VERBOSE)


def ensure_buildkit() -> None:
    result = subprocess.run(
        ["sudo", "systemctl", "is-active", "--quiet", "buildkit"],
        capture_output=True,
    )
    if result.returncode != 0:
        log.info("buildkit is not running, starting it...")
        subprocess.run(["sudo", "systemctl", "start", "buildkit"], check=True)
    # systemd may report "active" before buildkitd binds the unix socket
    # (e.g. after a containerd restart cascades through Requires=). Always wait.
    sock = "/run/buildkit/buildkitd.sock"
    for _ in range(20):
        if subprocess.run(["sudo", "test", "-S", sock]).returncode == 0:
            return
        time.sleep(0.5)
    raise RuntimeError(f"buildkit socket {sock} did not appear within 10s")
