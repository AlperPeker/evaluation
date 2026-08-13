import os

from shared.paths import now_ts


def _model_slug(model: str) -> str:
    return model.replace("/", "--")


def _image_slug(base_image: str) -> str:
    name = base_image.rsplit("/", 1)[-1]
    if ":" not in name:
        name += ":latest"
    return name.replace(":", "-")


# ── directory helpers ──────────────────────────────────────────────

def config_results_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "results", "config")

def config_charts_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "charts", "config")

def config_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(config_results_dir(base_dir), execution_ts)

def config_charts_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(config_charts_dir(base_dir), execution_ts)

def rebuild_results_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "results", "rebuild")

def rebuild_charts_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "charts", "rebuild")

def rebuild_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(rebuild_results_dir(base_dir), execution_ts)

def rebuild_charts_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(rebuild_charts_dir(base_dir), execution_ts)

def compression_results_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "results", "compression")

def compression_charts_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "charts", "compression")

# Layout: results/build/<ts>/{performance,resource}/  and  charts/build/<ts>/{performance,resource}/
# A single execution_ts folder bundles both subcategories.

def build_run_results_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(base_dir, "results", "build", execution_ts)

def build_run_charts_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(base_dir, "charts", "build", execution_ts)

def build_performance_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(build_run_results_dir(base_dir, execution_ts), "performance")

def build_performance_charts_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(build_run_charts_dir(base_dir, execution_ts), "performance")

def build_resource_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(build_run_results_dir(base_dir, execution_ts), "resource")

def build_resource_charts_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(build_run_charts_dir(base_dir, execution_ts), "resource")

def resource_cpu_charts_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(build_resource_charts_run_dir(base_dir, execution_ts), "cpu")

def resource_ram_charts_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(build_resource_charts_run_dir(base_dir, execution_ts), "ram")

def resource_cores_charts_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(build_resource_charts_run_dir(base_dir, execution_ts), "cores")

def resource_disk_charts_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(build_resource_charts_run_dir(base_dir, execution_ts), "disk")

def resource_net_charts_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(build_resource_charts_run_dir(base_dir, execution_ts), "net")

def build_artifacts_dir(base_dir: str, execution_ts: str, model: str, base_image: str, capacity: int) -> str:
    return os.path.join(
        base_dir, "artifacts", "build", execution_ts, "performance",
        f"{_model_slug(model)}_{_image_slug(base_image)}", f"cap_{capacity}",
    )

def rebuild_artifacts_dir(base_dir: str, execution_ts: str, model: str, base_image: str) -> str:
    return os.path.join(
        base_dir, "artifacts", "rebuild", execution_ts,
        f"{_model_slug(model)}_{_image_slug(base_image)}", "full",
    )


# ── output file paths ──────────────────────────────────────────────

def build_csv_path(base_dir: str, model: str, base_image: str, execution_ts: str) -> str:
    return os.path.join(build_performance_run_dir(base_dir, execution_ts), f"{_model_slug(model)}_{_image_slug(base_image)}.csv")

def build_chart_path(base_dir: str, model: str, base_image: str, execution_ts: str) -> str:
    return os.path.join(build_performance_charts_run_dir(base_dir, execution_ts), f"{_model_slug(model)}_{_image_slug(base_image)}_stages.png")

def resource_csv_path(base_dir: str, model: str, base_image: str, execution_ts: str) -> str:
    return os.path.join(build_resource_run_dir(base_dir, execution_ts), f"{_model_slug(model)}_{_image_slug(base_image)}_resource.csv")

def resource_chart_path(base_dir: str, model: str, base_image: str, execution_ts: str) -> str:
    return os.path.join(build_resource_charts_run_dir(base_dir, execution_ts), f"{_model_slug(model)}_{_image_slug(base_image)}_resource.png")

def rebuild_csv_path(base_dir: str, model: str, base_image: str, execution_ts: str) -> str:
    return os.path.join(rebuild_run_dir(base_dir, execution_ts), f"{_model_slug(model)}_{_image_slug(base_image)}_rebuild.csv")

def build_run_metadata_path(base_dir: str, execution_ts: str) -> str:
    return os.path.join(build_run_results_dir(base_dir, execution_ts), "run.json")

def rebuild_run_metadata_path(base_dir: str, execution_ts: str) -> str:
    return os.path.join(rebuild_run_dir(base_dir, execution_ts), "run.json")

def build_merged_csv_path(base_dir: str, execution_ts: str) -> str:
    return os.path.join(build_performance_run_dir(base_dir, execution_ts), "merged.csv")

def resource_merged_csv_path(base_dir: str, execution_ts: str) -> str:
    return os.path.join(build_resource_run_dir(base_dir, execution_ts), "merged.csv")

def rebuild_merged_csv_path(base_dir: str, execution_ts: str) -> str:
    return os.path.join(rebuild_run_dir(base_dir, execution_ts), "merged.csv")

def rebuild_multimodel_merged_csv_path(base_dir: str, execution_ts: str) -> str:
    return os.path.join(rebuild_run_dir(base_dir, execution_ts), "merged_multimodel.csv")

def rebuild_chart_path(base_dir: str, model: str, base_image: str, execution_ts: str) -> str:
    return os.path.join(rebuild_charts_run_dir(base_dir, execution_ts), f"{_model_slug(model)}_{_image_slug(base_image)}_rebuild.png")

def compression_csv_path(base_dir: str, model: str, base_image: str, max_splits: int) -> str:
    return os.path.join(compression_results_dir(base_dir), f"{_model_slug(model)}_{_image_slug(base_image)}_splits_{max_splits}_{now_ts()}.csv")

def compression_chart_path(base_dir: str, model: str, base_image: str, max_splits: int) -> str:
    return os.path.join(compression_charts_dir(base_dir), f"{_model_slug(model)}_{_image_slug(base_image)}_splits_{max_splits}_{now_ts()}.png")

def measure_config_csv_path(base_dir: str, model: str, base_image: str, mode: str, n_splits: int, execution_ts: str) -> str:
    mode_slug = mode.replace("-", "_")
    return os.path.join(config_run_dir(base_dir, execution_ts), f"{_model_slug(model)}_{_image_slug(base_image)}_{mode_slug}_splits_{n_splits}.csv")

def measure_config_chart_path(base_dir: str, model: str, base_image: str, mode: str, n_splits: int, execution_ts: str) -> str:
    mode_slug = mode.replace("-", "_")
    return os.path.join(config_charts_run_dir(base_dir, execution_ts), f"{_model_slug(model)}_{_image_slug(base_image)}_{mode_slug}_splits_{n_splits}.png")
