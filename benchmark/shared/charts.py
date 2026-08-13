MODE_COLORS: dict[str, str] = {
    "2dfs":                          "#1f77b4",
    "2dfs-stargz":                   "#ff7f0e",
    "2dfs-stargz-zstd":              "#9467bd",
    "2dfs-stargz-with-bg-fetch":     "#e377c2", # refresh-layer
    "2dfs-stargz-zstd-with-bg-fetch":"#8c564b", # refresh-layer
    "baseline-2dfs-stargz":          "#17becf",
    "stargz":                        "#2ca02c",
    "base":                          "#d62728",
}


def figure_footer(
    fig, model: str, base_image: str, fontsize: int = 8,
    max_allowed_splits: int | None = None,
    model_size: str | None = None,
    extra: str | None = None,
    lower_in: float = 0.0,
) -> None:
    """Stamp bottom-left corner of a figure with model/image metadata.

    lower_in shifts the footer text down by that many inches (chart-specific nudge).
    """
    text = f"model: {model}"
    if model_size is not None:
        text += f" ({model_size})"
    text += f"\nbase image: {base_image}"
    if max_allowed_splits is not None:
        text += f"\nmax_allowed_splits: {max_allowed_splits}"
    if extra is not None:
        text += f"\n{extra}"
    n_lines = text.count("\n") + 1
    fig_h_in = fig.get_size_inches()[1]
    footer_in = (n_lines * fontsize * 1.4) / 72.0
    pad_in = 0.35
    footer_block_frac = (footer_in + pad_in) / fig_h_in
    current_bottom = fig.subplotpars.bottom
    new_bottom = min(0.55, current_bottom + footer_block_frac)
    fig.subplots_adjust(bottom=new_bottom)
    fig.text(
        0.01, (footer_in + pad_in / 2 - lower_in) / fig_h_in,
        text,
        fontsize=fontsize,
        verticalalignment="top",
        family="monospace",
    )


def add_run_dots(ax, x_center: float, values: list[float], s: int = 12) -> None:
    """Scatter individual run dots with jitter around x_center."""
    for k, val in enumerate(values):
        jitter = (k - len(values) / 2) * 0.015
        ax.scatter(x_center + jitter, val, color="black", s=s, zorder=4)


def bar_group_xticks(
    ax, n_groups: int, n_modes: int, bar_width: float, labels: list[str]
) -> None:
    """Set centered xticks for grouped bar charts."""
    center_offset = (n_modes - 1) * bar_width / 2
    ax.set_xticks([j + center_offset for j in range(n_groups)])
    ax.set_xticklabels(labels)


def save_figure(fig, path: str, dpi: int = 150, log_path: bool = True) -> None:
    """savefig + close + log. Caller is responsible for tight_layout."""
    import matplotlib.pyplot as plt
    from shared import log
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    if log_path:
        log.result(f"Chart saved to {path}")


def write_csv(path: str, fieldnames: list[str], rows: list[dict]) -> None:
    """Write a DictWriter CSV to path, creating parent dirs. Logs filename."""
    import csv
    import os
    from shared import log
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.result(f"Results saved to {path}")
