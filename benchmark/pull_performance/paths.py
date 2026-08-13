import os


def _model_slug(model: str) -> str:
    return model.replace("/", "--")


def _image_slug(base_image: str) -> str:
    name = base_image.rsplit("/", 1)[-1]
    if ":" not in name:
        name += ":latest"
    return name.replace(":", "-")


# ── directory helpers ──────────────────────────────────────────────

def pull_results_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "results", "pull")

def pull_charts_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "charts", "pull")

def pull_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(pull_results_dir(base_dir), execution_ts)

def pull_charts_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(pull_charts_dir(base_dir), execution_ts)

def refresh_layer_results_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "results", "refresh-layer")

def refresh_layer_charts_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "charts", "refresh-layer")

def refresh_layer_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(refresh_layer_results_dir(base_dir), execution_ts)

def refresh_layer_charts_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(refresh_layer_charts_dir(base_dir), execution_ts)

def refresh_results_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "results", "refresh")

def refresh_charts_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "charts", "refresh")

def refresh_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(refresh_results_dir(base_dir), execution_ts)

def refresh_charts_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(refresh_charts_dir(base_dir), execution_ts)

def manual_update_results_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "results", "manual-update")

def manual_update_charts_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "charts", "manual-update")

def manual_update_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(manual_update_results_dir(base_dir), execution_ts)

def manual_update_charts_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(manual_update_charts_dir(base_dir), execution_ts)

def stargz_config_results_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "results", "stargz-config")

def stargz_config_charts_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "charts", "stargz-config")

def stargz_config_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(stargz_config_results_dir(base_dir), execution_ts)

def stargz_config_charts_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(stargz_config_charts_dir(base_dir), execution_ts)

def config_results_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "results", "config")

def config_charts_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "charts", "config")

def config_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(config_results_dir(base_dir), execution_ts)

def config_charts_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(config_charts_dir(base_dir), execution_ts)

def prefetch_layered_results_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "results", "prefetch-layered")

def prefetch_layered_charts_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "charts", "prefetch-layered")

def prefetch_layered_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(prefetch_layered_results_dir(base_dir), execution_ts)

def prefetch_layered_charts_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(prefetch_layered_charts_dir(base_dir), execution_ts)

def prefetch_stages_results_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "results", "prefetch-stages")

def prefetch_stages_charts_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "charts", "prefetch-stages")

def prefetch_stages_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(prefetch_stages_results_dir(base_dir), execution_ts)

def prefetch_stages_charts_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(prefetch_stages_charts_dir(base_dir), execution_ts)


# ── output file paths ──────────────────────────────────────────────

def pull_csv_path(base_dir: str, model: str, base_image: str, n_splits: int, execution_ts: str) -> str:
    return os.path.join(pull_run_dir(base_dir, execution_ts), f"{_model_slug(model)}_{_image_slug(base_image)}_pull_{n_splits}.csv")

def pull_chart_path(base_dir: str, model: str, base_image: str, n_splits: int, execution_ts: str) -> str:
    return os.path.join(pull_charts_run_dir(base_dir, execution_ts), f"{_model_slug(model)}_{_image_slug(base_image)}_pull_{n_splits}.png")

def pull_resource_csv_path(base_dir: str, model: str, base_image: str, execution_ts: str) -> str:
    return os.path.join(pull_run_dir(base_dir, execution_ts), "resource", f"{_model_slug(model)}_{_image_slug(base_image)}_resource.csv")

def pull_resource_merged_csv_path(base_dir: str, execution_ts: str) -> str:
    return os.path.join(pull_run_dir(base_dir, execution_ts), "resource", "merged.csv")

def pull_resource_charts_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(pull_charts_run_dir(base_dir, execution_ts), "resource")

def pull_resource_chart_path(base_dir: str, model: str, base_image: str, execution_ts: str) -> str:
    return os.path.join(pull_resource_charts_run_dir(base_dir, execution_ts), f"{_model_slug(model)}_{_image_slug(base_image)}_resource.png")

def pull_resource_cpu_charts_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(pull_resource_charts_run_dir(base_dir, execution_ts), "cpu")

def pull_resource_ram_charts_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(pull_resource_charts_run_dir(base_dir, execution_ts), "ram")

def pull_resource_cores_charts_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(pull_resource_charts_run_dir(base_dir, execution_ts), "cores")

def pull_resource_disk_charts_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(pull_resource_charts_run_dir(base_dir, execution_ts), "disk")

def pull_resource_net_charts_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(pull_resource_charts_run_dir(base_dir, execution_ts), "net")

def pull_stargz_config_path(base_dir: str, execution_ts: str) -> str:
    return os.path.join(pull_run_dir(base_dir, execution_ts), "stargz-config.toml")

def pull_merged_csv_path(base_dir: str, execution_ts: str) -> str:
    return os.path.join(pull_run_dir(base_dir, execution_ts), "merged.csv")

def pull_multimodel_csv_path(base_dir: str, label: str, base_image: str, n_models: int, execution_ts: str) -> str:
    return os.path.join(pull_run_dir(base_dir, execution_ts), f"{_model_slug(label)}_{_image_slug(base_image)}_multimodel_{n_models}.csv")

def pull_multimodel_chart_path(base_dir: str, label: str, base_image: str, n_models: int, execution_ts: str) -> str:
    return os.path.join(pull_charts_run_dir(base_dir, execution_ts), f"{_model_slug(label)}_{_image_slug(base_image)}_multimodel_{n_models}.png")

def pull_multimodel_merged_csv_path(base_dir: str, execution_ts: str) -> str:
    return os.path.join(pull_run_dir(base_dir, execution_ts), "merged_multimodel.csv")

def pull_multimodel_resource_merged_csv_path(base_dir: str, execution_ts: str) -> str:
    return os.path.join(pull_run_dir(base_dir, execution_ts), "resource", "merged_multimodel.csv")

def pull_run_metadata_path(base_dir: str, execution_ts: str) -> str:
    return os.path.join(pull_run_dir(base_dir, execution_ts), "run.json")

def refresh_run_metadata_path(base_dir: str, execution_ts: str) -> str:
    return os.path.join(refresh_run_dir(base_dir, execution_ts), "run.json")

def refresh_layer_csv_path(base_dir: str, model: str, base_image: str, execution_ts: str) -> str:
    return os.path.join(refresh_layer_run_dir(base_dir, execution_ts), f"{_model_slug(model)}_{_image_slug(base_image)}_refresh_layer.csv")

def refresh_layer_chart_path(base_dir: str, model: str, base_image: str, execution_ts: str) -> str:
    return os.path.join(refresh_layer_charts_run_dir(base_dir, execution_ts), f"{_model_slug(model)}_{_image_slug(base_image)}_refresh_layer.png")

def refresh_csv_path(base_dir: str, model: str, base_image: str, execution_ts: str) -> str:
    return os.path.join(refresh_run_dir(base_dir, execution_ts), f"{_model_slug(model)}_{_image_slug(base_image)}_refresh.csv")

def refresh_chart_path(base_dir: str, model: str, base_image: str, execution_ts: str) -> str:
    return os.path.join(refresh_charts_run_dir(base_dir, execution_ts), f"{_model_slug(model)}_{_image_slug(base_image)}_refresh.png")

def refresh_bytes_csv_path(base_dir: str, model: str, base_image: str, execution_ts: str) -> str:
    return os.path.join(refresh_run_dir(base_dir, execution_ts), f"{_model_slug(model)}_{_image_slug(base_image)}_refresh_bytes.csv")

def refresh_bytes_chart_path(base_dir: str, model: str, base_image: str, execution_ts: str) -> str:
    return os.path.join(refresh_charts_run_dir(base_dir, execution_ts), f"{_model_slug(model)}_{_image_slug(base_image)}_refresh_bytes.png")

def refresh_merged_csv_path(base_dir: str, execution_ts: str) -> str:
    return os.path.join(refresh_run_dir(base_dir, execution_ts), "merged.csv")

def refresh_merged_bytes_csv_path(base_dir: str, execution_ts: str) -> str:
    return os.path.join(refresh_run_dir(base_dir, execution_ts), "merged_bytes.csv")

def refresh_multimodel_merged_csv_path(base_dir: str, execution_ts: str) -> str:
    return os.path.join(refresh_run_dir(base_dir, execution_ts), "merged_multimodel.csv")

def refresh_multimodel_merged_bytes_csv_path(base_dir: str, execution_ts: str) -> str:
    return os.path.join(refresh_run_dir(base_dir, execution_ts), "merged_multimodel_bytes.csv")

def refresh_stargz_config_path(base_dir: str, execution_ts: str) -> str:
    return os.path.join(refresh_run_dir(base_dir, execution_ts), "stargz-config.toml")

def manual_update_csv_path(base_dir: str, model: str, base_image: str, execution_ts: str) -> str:
    return os.path.join(manual_update_run_dir(base_dir, execution_ts), f"{_model_slug(model)}_{_image_slug(base_image)}_manual_update.csv")

def manual_update_chart_path(base_dir: str, model: str, base_image: str, execution_ts: str) -> str:
    return os.path.join(manual_update_charts_run_dir(base_dir, execution_ts), f"{_model_slug(model)}_{_image_slug(base_image)}_manual_update.png")

def stargz_config_csv_path(base_dir: str, model: str, base_image: str, execution_ts: str) -> str:
    return os.path.join(stargz_config_run_dir(base_dir, execution_ts), f"{_model_slug(model)}_{_image_slug(base_image)}.csv")

def stargz_config_chart_path(base_dir: str, model: str, base_image: str, mode: str, execution_ts: str) -> str:
    mode_slug = mode.replace("-", "_")
    return os.path.join(stargz_config_charts_run_dir(base_dir, execution_ts), f"{_model_slug(model)}_{_image_slug(base_image)}_{mode_slug}.png")

def stargz_config_merged_csv_path(base_dir: str, execution_ts: str) -> str:
    return os.path.join(stargz_config_run_dir(base_dir, execution_ts), "merged.csv")

def stargz_config_base_config_path(base_dir: str, execution_ts: str) -> str:
    return os.path.join(stargz_config_run_dir(base_dir, execution_ts), "stargz-config.toml")

def stargz_config_run_metadata_path(base_dir: str, execution_ts: str) -> str:
    return os.path.join(stargz_config_run_dir(base_dir, execution_ts), "run.json")

def build_config_csv_path(base_dir: str, model: str, base_image: str, mode: str, execution_ts: str) -> str:
    mode_slug = mode.replace("-", "_")
    return os.path.join(config_run_dir(base_dir, execution_ts), f"{_model_slug(model)}_{_image_slug(base_image)}_{mode_slug}.csv")

def build_config_chart_path(base_dir: str, model: str, base_image: str, mode: str, execution_ts: str) -> str:
    mode_slug = mode.replace("-", "_")
    return os.path.join(config_charts_run_dir(base_dir, execution_ts), f"{_model_slug(model)}_{_image_slug(base_image)}_{mode_slug}.png")

def prefetch_layered_csv_path(base_dir: str, model: str, base_image: str, execution_ts: str) -> str:
    return os.path.join(prefetch_layered_run_dir(base_dir, execution_ts), f"{_model_slug(model)}_{_image_slug(base_image)}.csv")

def prefetch_layered_chart_path(base_dir: str, model: str, base_image: str, mode: str, execution_ts: str) -> str:
    mode_slug = mode.replace("-", "_")
    return os.path.join(prefetch_layered_charts_run_dir(base_dir, execution_ts), f"{_model_slug(model)}_{_image_slug(base_image)}_{mode_slug}.png")

def prefetch_stages_csv_path(base_dir: str, model: str, base_image: str, execution_ts: str) -> str:
    return os.path.join(prefetch_stages_run_dir(base_dir, execution_ts), f"{_model_slug(model)}_{_image_slug(base_image)}.csv")

def prefetch_stages_chart_path(base_dir: str, model: str, base_image: str, mode: str, execution_ts: str) -> str:
    mode_slug = mode.replace("-", "_")
    return os.path.join(prefetch_stages_charts_run_dir(base_dir, execution_ts), f"{_model_slug(model)}_{_image_slug(base_image)}_{mode_slug}.png")

def prefetch_stages_merged_csv_path(base_dir: str, execution_ts: str) -> str:
    return os.path.join(prefetch_stages_run_dir(base_dir, execution_ts), "merged.csv")

def prefetch_stages_run_metadata_path(base_dir: str, execution_ts: str) -> str:
    return os.path.join(prefetch_stages_run_dir(base_dir, execution_ts), "run.json")

def prefetch_stages_base_config_path(base_dir: str, execution_ts: str) -> str:
    return os.path.join(prefetch_stages_run_dir(base_dir, execution_ts), "stargz-config.toml")

def prefetch_layered_logs_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "logs", "prefetch-layered")

def prefetch_layered_logs_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(prefetch_layered_logs_dir(base_dir), execution_ts)

def prefetch_layered_log_path(base_dir: str, model: str, base_image: str, mode: str, n: int, execution_ts: str) -> str:
    mode_slug = mode.replace("-", "_")
    return os.path.join(prefetch_layered_logs_run_dir(base_dir, execution_ts), f"{_model_slug(model)}_{_image_slug(base_image)}_{mode_slug}_{n}allotments.json")

def stargz_config_logs_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "logs", "stargz-config")

def stargz_config_logs_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(stargz_config_logs_dir(base_dir), execution_ts)

def stargz_config_log_path(base_dir: str, model: str, base_image: str, mode: str, config_label: str, n: int, run: int, execution_ts: str) -> str:
    mode_slug = mode.replace("-", "_")
    label_slug = config_label.replace(" ", "_").replace(",", "").replace("/", "_")
    return os.path.join(stargz_config_logs_run_dir(base_dir, execution_ts), f"{_model_slug(model)}_{_image_slug(base_image)}_{mode_slug}_{label_slug}_{n}allotments_run{run}.json")

def refresh_layer_logs_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "logs", "refresh-layer")

def refresh_layer_logs_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(refresh_layer_logs_dir(base_dir), execution_ts)

def refresh_layer_log_path(base_dir: str, model: str, base_image: str, arm: str, run: int, execution_ts: str) -> str:
    return os.path.join(refresh_layer_logs_run_dir(base_dir, execution_ts), f"{_model_slug(model)}_{_image_slug(base_image)}_{arm}_run{run}.json")

def refresh_logs_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "logs", "refresh")

def refresh_logs_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(refresh_logs_dir(base_dir), execution_ts)

def refresh_log_path(base_dir: str, model: str, base_image: str, arm: str, run: int, execution_ts: str) -> str:
    return os.path.join(refresh_logs_run_dir(base_dir, execution_ts), f"{_model_slug(model)}_{_image_slug(base_image)}_{arm}_run{run}.json")


def prefetch_stages_logs_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "logs", "prefetch-stages")

def prefetch_stages_logs_run_dir(base_dir: str, execution_ts: str) -> str:
    return os.path.join(prefetch_stages_logs_dir(base_dir), execution_ts)

def prefetch_stages_log_path(base_dir: str, model: str, base_image: str, mode: str, config_label: str, n: int, run: int, execution_ts: str) -> str:
    mode_slug = mode.replace("-", "_")
    label_slug = config_label.replace(" ", "_").replace(",", "").replace("/", "_")
    return os.path.join(prefetch_stages_logs_run_dir(base_dir, execution_ts), f"{_model_slug(model)}_{_image_slug(base_image)}_{mode_slug}_{label_slug}_{n}allotments_run{run}.json")


# ── artifact directories ──────────────────────────────────────────

def _experiment_dir(base_dir: str, scope: str, execution_ts: str, model: str, base_image: str) -> str:
    return os.path.join(
        base_dir, "artifacts", scope, execution_ts,
        f"{_model_slug(model)}_{_image_slug(base_image)}",
    )

def pull_artifacts_dir(base_dir: str, execution_ts: str, model: str, base_image: str, mode: str, n: int | None = None) -> str:
    root = _experiment_dir(base_dir, "pull", execution_ts, model, base_image)
    if mode == "base":
        if n is None:
            raise ValueError("n required for base mode")
        return os.path.join(root, "base", f"n_{n}")
    return os.path.join(root, mode)

def refresh_layer_artifacts_dir(base_dir: str, execution_ts: str, model: str, base_image: str, build_mode: str) -> str:
    return os.path.join(_experiment_dir(base_dir, "refresh-layer", execution_ts, model, base_image), build_mode)

def refresh_artifacts_dir(base_dir: str, execution_ts: str, model: str, base_image: str, build_mode: str) -> str:
    return os.path.join(_experiment_dir(base_dir, "refresh", execution_ts, model, base_image), build_mode)

def manual_update_artifacts_dir(base_dir: str, execution_ts: str, model: str, base_image: str, build_mode: str) -> str:
    return os.path.join(_experiment_dir(base_dir, "manual-update", execution_ts, model, base_image), build_mode)

def build_config_artifacts_dir(base_dir: str, execution_ts: str, model: str, base_image: str, label: str) -> str:
    label_slug = label.replace(" ", "_").replace(",", "").replace("/", "_")
    return os.path.join(_experiment_dir(base_dir, "build-config", execution_ts, model, base_image), label_slug)

def stargz_config_artifacts_dir(base_dir: str, execution_ts: str, model: str, base_image: str, mode: str) -> str:
    return os.path.join(_experiment_dir(base_dir, "stargz-config", execution_ts, model, base_image), mode)

def prefetch_layered_artifacts_dir(base_dir: str, execution_ts: str, model: str, base_image: str, mode: str) -> str:
    return os.path.join(_experiment_dir(base_dir, "prefetch-layered", execution_ts, model, base_image), mode)

def prefetch_stages_artifacts_dir(base_dir: str, execution_ts: str, model: str, base_image: str, mode: str) -> str:
    return os.path.join(_experiment_dir(base_dir, "prefetch-stages", execution_ts, model, base_image), mode)
