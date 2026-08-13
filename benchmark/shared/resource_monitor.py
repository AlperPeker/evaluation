import csv
import os
import threading
import time
from dataclasses import asdict, dataclass, fields

import psutil

from shared import log
from shared.fs import network_interface, physical_device

RESOURCE_SCHEMA_VERSION = 6


@dataclass(frozen=True)
class ResourceRow:
    """One sample. Disk *_bytes/_count/_time counters are RAW cumulative-since-boot
    (rates/util derived later via derive_samples); disk_used_bytes is the data-volume
    space used at sample time (absolute value)."""
    schema_version: int
    model: str
    base_image: str
    max_allowed_splits: int
    timestamp_ms: int
    cpu_percent: float
    cpu_per_core: str
    mem_mb: float
    disk_read_bytes: int
    disk_write_bytes: int
    disk_read_count: int
    disk_write_count: int
    disk_read_time_ms: int
    disk_write_time_ms: int
    disk_busy_time_ms: int
    disk_used_bytes: int  # data-volume space used right now
    disk_total_bytes: int  # data-volume capacity (constant per node; lets free/percent be derived)
    net_recv_bytes: int  # registry-facing NIC only (see network_interface)
    net_sent_bytes: int
    mode: str
    dimension: int | None  # capacity (build) / partition_pct (pull); CSV column renamed at write
    run: int | None


@dataclass
class DiskRates:
    """Rates derived from the delta between two consecutive raw ResourceRow counters."""
    read_mb_s: float = 0.0
    write_mb_s: float = 0.0
    read_iops: float = 0.0
    write_iops: float = 0.0
    util_pct: float = 0.0  # busy_time -> wall fraction the device was active (<=100%)


@dataclass
class NetRates:
    """Rates derived from the delta between two consecutive raw ResourceRow counters."""
    recv_mb_s: float = 0.0
    sent_mb_s: float = 0.0


@dataclass
class DerivedSample:
    """A raw ResourceRow paired with disk/net rates derived against the previous sample."""
    row: ResourceRow
    disk: DiskRates
    net: NetRates


def derive_samples(samples: list[ResourceRow]) -> list[DerivedSample]:
    """Attach per-window disk rates to each row, deriving from consecutive raw
    counters over the full continuous capture (so no window loses its first point)."""
    ordered = sorted(samples, key=lambda r: r.timestamp_ms)
    out: list[DerivedSample] = []
    prev: ResourceRow | None = None
    for row in ordered:
        disk = DiskRates()
        net = NetRates()
        if prev is not None:
            dt = (row.timestamp_ms - prev.timestamp_ms) / 1000.0
            if dt > 0:
                def rate(attr: str) -> float:
                    return (getattr(row, attr) - getattr(prev, attr)) / dt
                disk = DiskRates(
                    read_mb_s=rate("disk_read_bytes") / (1024 * 1024),
                    write_mb_s=rate("disk_write_bytes") / (1024 * 1024),
                    read_iops=rate("disk_read_count"),
                    write_iops=rate("disk_write_count"),
                    util_pct=rate("disk_busy_time_ms") / 1000 * 100,
                )
                net = NetRates(
                    recv_mb_s=rate("net_recv_bytes") / (1024 * 1024),
                    sent_mb_s=rate("net_sent_bytes") / (1024 * 1024),
                )
        out.append(DerivedSample(row=row, disk=disk, net=net))
        prev = row
    return out


class ResourceMonitor:
    """Threaded CPU/RAM/disk sampler. Records RAW cumulative disk counters (rates/util
    derived later via derive_samples), stamping each sample with the (mode, dimension, run)
    context set by the caller. `dimension` is an opaque bucket (capacity / partition_pct)."""

    def __init__(self, model: str, base_image: str, max_allowed_splits: int, tmpdir: str,
                 data_volume: str = "/", registry_host: str | None = None):
        self._samples: list[ResourceRow] = []
        self._model = model
        self._base_image = base_image
        self._max_allowed_splits = max_allowed_splits
        self._mode: str = "idle"
        self._dimension: int | None = None
        self._run: int | None = None
        self._stop = threading.Event()
        self._data_volume = data_volume
        # work stages through this TMPDIR; sample the physical disk backing it
        self._disk_dev = physical_device(tmpdir)
        if self._disk_dev:
            log.info(f"Disk metrics sampling physical device: {self._disk_dev}")
        else:
            log.info("Disk device detection failed; disk util% is whole-system sum (may exceed 100%)")
        # registry traffic rides one NIC; sample it so net excludes mgmt/docker-bridge noise
        self._net_iface = network_interface(registry_host) if registry_host else None
        if self._net_iface:
            log.info(f"Network metrics sampling interface: {self._net_iface}")
        else:
            log.info("Network interface detection failed; net counters are whole-system sum")

    def _disk_counters(self):
        if self._disk_dev:
            return psutil.disk_io_counters(perdisk=True).get(self._disk_dev)
        return psutil.disk_io_counters()

    def _net_counters(self):
        if self._net_iface:
            return psutil.net_io_counters(pernic=True).get(self._net_iface)
        return psutil.net_io_counters()

    def set_context(self, mode: str, dimension: int, run: int) -> None:
        self._mode = mode
        self._dimension = dimension
        self._run = run

    def set_idle(self) -> None:
        self._mode = "idle"
        self._dimension = None
        self._run = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> list[ResourceRow]:
        self._stop.set()
        self._thread.join()
        return self._samples

    def _poll(self) -> None:
        while not self._stop.is_set():
            # percpu sample blocks for the interval; aggregate cpu = mean of cores
            per_core = psutil.cpu_percent(interval=1, percpu=True)
            cpu = sum(per_core) / len(per_core) if per_core else 0.0
            mem = psutil.virtual_memory().used / (1024 * 1024)  # MB

            # record RAW cumulative counters; rates/util/await derived at plot time
            disk = self._disk_counters()
            disk_usage = psutil.disk_usage(self._data_volume)
            net = self._net_counters()
            ts = int(time.time() * 1000)
            self._samples.append(ResourceRow(
                schema_version=RESOURCE_SCHEMA_VERSION,
                model=self._model,
                base_image=self._base_image,
                max_allowed_splits=self._max_allowed_splits,
                timestamp_ms=ts,
                cpu_percent=cpu,
                cpu_per_core="|".join(f"{c:.1f}" for c in per_core),
                mem_mb=mem,
                disk_read_bytes=disk.read_bytes if disk else 0,
                disk_write_bytes=disk.write_bytes if disk else 0,
                disk_read_count=disk.read_count if disk else 0,
                disk_write_count=disk.write_count if disk else 0,
                disk_read_time_ms=disk.read_time if disk else 0,
                disk_write_time_ms=disk.write_time if disk else 0,
                disk_busy_time_ms=disk.busy_time if disk else 0,
                disk_used_bytes=disk_usage.used,
                disk_total_bytes=disk_usage.total,
                net_recv_bytes=net.bytes_recv if net else 0,
                net_sent_bytes=net.bytes_sent if net else 0,
                mode=self._mode,
                dimension=self._dimension,
                run=self._run,
            ))


def write_resource_csv(output_path: str, samples: list[ResourceRow], dimension_col: str = "dimension") -> None:
    """Write raw rows, renaming the generic `dimension` column to the caller's name."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [dimension_col if f.name == "dimension" else f.name for f in fields(ResourceRow)]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples:
            row = asdict(s)
            dim = row.pop("dimension")
            row[dimension_col] = "" if dim is None else dim
            if row["run"] is None:
                row["run"] = ""
            writer.writerow(row)
    log.result(f"Resource CSV saved to {output_path}")
