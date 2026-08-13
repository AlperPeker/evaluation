import os
from collections import defaultdict
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

from shared.charts import MODE_COLORS, bar_group_xticks, figure_footer, save_figure
from shared.registry import image_slug
from shared.resource_monitor import DerivedSample


@dataclass
class RunSamples:
    """Derived per-window rates collected during one (dimension, mode, run) window."""
    cpu: list[float] = field(default_factory=list)
    mem: list[float] = field(default_factory=list)
    disk_read_mb: list[float] = field(default_factory=list)
    disk_write_mb: list[float] = field(default_factory=list)
    disk_read_iops: list[float] = field(default_factory=list)
    disk_write_iops: list[float] = field(default_factory=list)
    disk_util: list[float] = field(default_factory=list)
    net_recv_mb: list[float] = field(default_factory=list)
    net_sent_mb: list[float] = field(default_factory=list)


def _bar_stats(by_dim_mode, dimensions, mode: str, attr: str):
    """Per-dimension (mean-of-run-medians, std) for one RunSamples attribute."""
    heights, errs = [], []
    for dim in dimensions:
        runs = by_dim_mode.get((dim, mode), {})
        medians = [float(np.median(getattr(rs, attr))) for rs in runs.values() if getattr(rs, attr)]
        heights.append(float(np.mean(medians)) if medians else 0.0)
        errs.append(float(np.std(medians, ddof=0)) if medians else 0.0)
    return heights, errs


def _grouped_bars(ax, x, dimensions, x_labels, series, ylabel,
                  xlabel=None, ylim=None, title=None) -> None:
    """series: list of dicts with keys label, color, hatch, heights, errs."""
    n = len(series)
    width = 0.8 / n
    for i, s in enumerate(series):
        offsets = [pos + i * width for pos in x]
        ax.bar(offsets, s["heights"], width, yerr=s["errs"], label=s["label"],
               color=s["color"], hatch=s["hatch"], edgecolor="black", linewidth=0.5,
               error_kw={"ecolor": "black", "capsize": 3, "elinewidth": 1})
    bar_group_xticks(ax, len(dimensions), n, width, x_labels)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylim:
        ax.set_ylim(*ylim)
    if title:
        ax.set_title(title)
    ax.legend(fontsize="small")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")


def plot_resource_aggregate(
    samples: list[DerivedSample], *,
    modes: list[str], n_runs: int, xlabel: str, title: str,
    model: str, base_image: str, max_allowed_splits: int, output_path: str,
) -> None:
    """Aggregate bar chart: CPU, mem, disk throughput, disk util (total), disk IOPS."""
    if not samples:
        return

    colors = {mode: MODE_COLORS[mode] for mode in modes}

    # (dimension, mode) -> {run_index: RunSamples}
    by_dim_mode: dict[tuple[int, str], dict[int, RunSamples]] = defaultdict(
        lambda: defaultdict(RunSamples)
    )

    for s in samples:
        r = s.row
        if r.mode == "idle" or r.dimension is None or r.run is None:
            continue
        bucket = by_dim_mode[(r.dimension, r.mode)][r.run]
        bucket.cpu.append(r.cpu_percent)
        bucket.mem.append(r.mem_mb)
        bucket.disk_read_mb.append(s.disk.read_mb_s)
        bucket.disk_write_mb.append(s.disk.write_mb_s)
        bucket.disk_read_iops.append(s.disk.read_iops)
        bucket.disk_write_iops.append(s.disk.write_iops)
        bucket.disk_util.append(s.disk.util_pct)
        bucket.net_recv_mb.append(s.net.recv_mb_s)
        bucket.net_sent_mb.append(s.net.sent_mb_s)

    dimensions = sorted({dim for (dim, _) in by_dim_mode.keys()})
    if not dimensions:
        return

    x_labels = [f"{d}" for d in dimensions]
    x = range(len(dimensions))

    # read = solid fill, write = "//" hatch; color encodes mode. util = total only.
    cpu_series, mem_series, tput_series, util_series, iops_series, net_series = [], [], [], [], [], []
    for mode in modes:
        color = colors[mode]

        def stat(attr: str):
            return _bar_stats(by_dim_mode, dimensions, mode, attr)

        def entry(label, attr, hatch=None):
            heights, errs = stat(attr)
            return {"label": label, "color": color, "hatch": hatch, "heights": heights, "errs": errs}

        cpu_series.append(entry(mode, "cpu"))
        mem_series.append(entry(mode, "mem"))
        tput_series.append(entry(f"{mode} read", "disk_read_mb"))
        tput_series.append(entry(f"{mode} write", "disk_write_mb", hatch="//"))
        util_series.append(entry(f"{mode} total", "disk_util"))
        iops_series.append(entry(f"{mode} read", "disk_read_iops"))
        iops_series.append(entry(f"{mode} write", "disk_write_iops", hatch="//"))
        net_series.append(entry(f"{mode} recv", "net_recv_mb"))
        net_series.append(entry(f"{mode} sent", "net_sent_mb", hatch="//"))

    fig, (ax_cpu, ax_mem, ax_tput, ax_util, ax_iops, ax_net) = plt.subplots(
        6, 1, figsize=(max(8, len(dimensions) * 2), 21))

    _grouped_bars(ax_cpu, x, dimensions, x_labels, cpu_series, "CPU Usage (%)", title=title)
    _grouped_bars(ax_mem, x, dimensions, x_labels, mem_series, "Memory Usage (MB)")
    _grouped_bars(ax_tput, x, dimensions, x_labels, tput_series, "Disk throughput (MB/s)")
    _grouped_bars(ax_util, x, dimensions, x_labels, util_series, "Disk util (%)")
    _grouped_bars(ax_iops, x, dimensions, x_labels, iops_series, "Disk IOPS (ops/s)")
    _grouped_bars(ax_net, x, dimensions, x_labels, net_series, "Network (MB/s)", xlabel=xlabel)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    figure_footer(fig, model, base_image, max_allowed_splits=max_allowed_splits)
    save_figure(fig, output_path)


def plot_resource_timeseries(
    samples: list[DerivedSample], *,
    model: str, base_image: str, max_allowed_splits: int,
    dimension_label: str, dimension_tag: str,
    cpu_dir: str, ram_dir: str, cores_dir: str, disk_dir: str, net_dir: str,
    dimension_unit: str = "%",
) -> None:
    """Per-(mode, dimension, run) over-time charts: CPU, RAM, per-core heatmap, disk, net."""
    if not samples:
        return

    series: dict[tuple[str, int, int], list[DerivedSample]] = defaultdict(list)

    def _per_core(per_core: str) -> list[float]:
        return [float(c) for c in per_core.split("|") if c]

    for s in samples:
        r = s.row
        if r.mode == "idle" or r.dimension is None or r.run is None:
            continue
        series[(r.mode, r.dimension, r.run)].append(s)

    model_slug = model.replace("/", "--")
    img_slug = image_slug(base_image)
    for d in (cpu_dir, ram_dir, cores_dir, disk_dir, net_dir):
        os.makedirs(d, exist_ok=True)

    for (mode_name, dim, run), samps in sorted(series.items()):
        samps.sort(key=lambda s: s.row.timestamp_ms)
        rows = [s.row for s in samps]
        t0 = rows[0].timestamp_ms
        t_sec = [(r.timestamp_ms - t0) / 1000.0 for r in rows]
        cpu_vals = [r.cpu_percent for r in rows]
        mem_vals = [r.mem_mb for r in rows]
        per_core_rows = [_per_core(r.cpu_per_core) for r in rows]

        mode_slug = mode_name.replace("-", "_")
        file_stem = f"{model_slug}_{img_slug}_{mode_slug}_run{run + 1}_{dimension_tag}{dim}"

        def _add_run_footer(fig) -> None:
            figure_footer(fig, model, base_image, max_allowed_splits=max_allowed_splits)
            fig.text(
                0.99, 0.01,
                f"mode: {mode_name}  |  run: {run + 1}  |  {dimension_label}: {dim}{dimension_unit}",
                fontsize=8,
                verticalalignment="bottom",
                horizontalalignment="right",
                family="monospace",
            )

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(t_sec, cpu_vals, color=MODE_COLORS.get(mode_name, "#888888"), linewidth=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("CPU (%)")
        ax.set_title("CPU usage over time")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        _add_run_footer(fig)
        save_figure(fig, os.path.join(cpu_dir, f"{file_stem}.png"), log_path=False)

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(t_sec, mem_vals, color=MODE_COLORS.get(mode_name, "#888888"), linewidth=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Memory (MB)")
        ax.set_title("RAM usage over time")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        _add_run_footer(fig)
        save_figure(fig, os.path.join(ram_dir, f"{file_stem}.png"), log_path=False)

        # per-core heatmap: rows = cores, cols = time; truncate to common core count
        ncores = min((len(r) for r in per_core_rows), default=0)
        if ncores > 0 and len(t_sec) > 1:
            heat = np.array([r[:ncores] for r in per_core_rows]).T  # [core, time]
            fig, ax = plt.subplots(figsize=(8, max(3, ncores * 0.18)))
            im = ax.imshow(
                heat, aspect="auto", origin="lower", cmap="inferno",
                vmin=0, vmax=100,
                extent=(t_sec[0], t_sec[-1], -0.5, ncores - 0.5),
            )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Core index")
            ax.set_title("Per-core CPU usage over time")
            fig.colorbar(im, ax=ax, label="CPU (%)")
            fig.tight_layout()
            _add_run_footer(fig)
            save_figure(fig, os.path.join(cores_dir, f"{file_stem}.png"), log_path=False)

        # disk over time: throughput (read/write), util (total only), IOPS (read/write)
        read_c, write_c, total_c = "#1f77b4", "#d62728", "#999999"
        disk = [s.disk for s in samps]
        fig, (ax_t, ax_u, ax_i) = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
        ax_t.plot(t_sec, [d.read_mb_s for d in disk], color=read_c, linewidth=1, label="read")
        ax_t.plot(t_sec, [d.write_mb_s for d in disk], color=write_c, linewidth=1, label="write")
        ax_t.set_ylabel("Throughput (MB/s)")
        ax_t.set_title("Disk activity over time")

        ax_u.plot(t_sec, [d.util_pct for d in disk], color=total_c, linewidth=1.5, label="total")
        ax_u.set_ylabel("Util (%)")

        ax_i.plot(t_sec, [d.read_iops for d in disk], color=read_c, linewidth=1, label="read")
        ax_i.plot(t_sec, [d.write_iops for d in disk], color=write_c, linewidth=1, label="write")
        ax_i.set_ylabel("IOPS (ops/s)")
        ax_i.set_xlabel("Time (s)")

        for axis in (ax_t, ax_u, ax_i):
            axis.grid(True, linestyle="--", alpha=0.5)
            axis.legend(fontsize="small")
        fig.tight_layout()
        _add_run_footer(fig)
        save_figure(fig, os.path.join(disk_dir, f"{file_stem}.png"), log_path=False)

        # net over time: recv is the lazy-load signal on the registry-facing NIC
        net = [s.net for s in samps]
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(t_sec, [n.recv_mb_s for n in net], color=read_c, linewidth=1, label="recv")
        ax.plot(t_sec, [n.sent_mb_s for n in net], color=write_c, linewidth=1, label="sent")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Network (MB/s)")
        ax.set_title("Network activity over time")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize="small")
        fig.tight_layout()
        _add_run_footer(fig)
        save_figure(fig, os.path.join(net_dir, f"{file_stem}.png"), log_path=False)
