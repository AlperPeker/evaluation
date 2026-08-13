import os
import shutil
import subprocess


def network_interface(host: str) -> str | None:
    """Resolve the NIC the kernel routes to `host` (host may carry a :port).

    Asks the routing table (`ip route get`) so network metrics sample only the
    interface carrying registry traffic, not the control/mgmt NIC or docker bridge.
    """
    ip = host.partition(":")[0]
    try:
        out = subprocess.run(
            ["ip", "route", "get", ip], capture_output=True, text=True, timeout=5
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    tokens = out.split()
    return tokens[tokens.index("dev") + 1] if "dev" in tokens else None


def physical_device(path: str) -> str | None:
    """Resolve the leaf physical block device backing `path`.

    Walks the LVM/partition stack (e.g. dm-0 -> sda2 -> sda) so disk metrics
    sample a single device instead of summing every layer (which triple-counts
    the same I/O and pushes util% past 100% on device-mapper hosts).
    """
    try:
        st = os.stat(path)
        dev = os.path.basename(
            os.path.realpath(f"/sys/dev/block/{os.major(st.st_dev)}:{os.minor(st.st_dev)}"))
    except OSError:
        return None
    seen: set[str] = set()
    while dev and dev not in seen:
        seen.add(dev)
        slaves_dir = f"/sys/class/block/{dev}/slaves"
        slaves = os.listdir(slaves_dir) if os.path.isdir(slaves_dir) else []
        if slaves:  # dm/md device -> descend to underlying device
            dev = slaves[0]
            continue
        if os.path.exists(f"/sys/class/block/{dev}/partition"):  # partition -> parent disk
            parent = os.path.basename(os.path.dirname(os.path.realpath(f"/sys/class/block/{dev}")))
            if parent and parent != dev:
                dev = parent
                continue
        break
    return dev or None


def rmtree(path: str) -> None:
    """Recursively delete a directory. No-op if it doesn't exist."""
    shutil.rmtree(path, ignore_errors=True)


def clear_dir(path: str) -> None:
    """Delete all contents of a directory, recreating it empty."""
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
