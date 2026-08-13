import json
import os
from urllib.parse import urlencode
from urllib.request import urlopen

from shared import log

PROM_URL = os.environ.get("PROM_URL", "http://127.0.0.1:9090")
HTTP_TIMEOUT_S = 10

_warned = False


def is_alive() -> bool:
    try:
        with urlopen(f"{PROM_URL}/-/healthy", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def _warn_once_if_dead() -> bool:
    """Emit a single warning if Prometheus is unreachable. Returns True if alive."""
    global _warned
    if is_alive():
        return True
    if not _warned:
        log.info(f"  WARNING: Prometheus not reachable at {PROM_URL}. "
                 f"Start it with local/start-prometheus.sh. Returning empty results.")
        _warned = True
    return False


def query(promql: str, t: float | None = None) -> list[dict]:
    """Instant query at time `t` (epoch seconds). Returns the `result` array, or []."""
    if not _warn_once_if_dead():
        return []
    params = {"query": promql}
    if t is not None:
        params["time"] = f"{t}"
    return _get(f"/api/v1/query?{urlencode(params)}")


def query_range(promql: str, start_s: float, end_s: float, step: str = "500ms") -> list[dict]:
    """Range query. Returns the `result` array (each entry has `metric` and `values`), or []."""
    if not _warn_once_if_dead():
        return []
    params = urlencode({"query": promql, "start": f"{start_s}", "end": f"{end_s}", "step": step})
    return _get(f"/api/v1/query_range?{params}")


def active_window(
    promql: str, start_s: float, end_s: float,
    step: str = "500ms", widen_back_s: float = 60.0,
    target_up_query: str | None = None,
) -> tuple[float, float] | None:
    """First and last timestamp at which `promql` (a counter) had activity inside [start_s, end_s].

    The query is widened back by `widen_back_s` to make pre-window state visible.

    Counter resets are detected two ways:
      1. an explicit drop (v < prev) in `promql` itself, OR
      2. a 0→1 transition in `target_up_query` (the target's `up` metric),
         which catches the case where the target restarts and all activity
         fits inside one scrape interval — leaving no observable drop in the
         counter, only an "appears at value > 0" jump.

    Activity is recorded when:
      - a sample's value is > the previous sample's value, OR
      - a reset is detected (either kind) and the post-reset value is > 0,
      - the first sample seen post-reset (or in absence of any baseline) is > 0.

    Only timestamps within [start_s, end_s] count; pre-window samples only
    inform restart/reset detection.

    Returns (first_active_ts, last_active_ts), or None if no activity.
    """
    res = query_range(promql, start_s - widen_back_s, end_s, step)
    if not res:
        return None
    values = [(float(t), float(v)) for t, v in res[0]["values"]]
    if not values:
        return None

    restarts: list[float] = []
    if target_up_query:
        up_res = query_range(target_up_query, start_s - widen_back_s, end_s, step)
        if up_res:
            up_values = [(float(t), float(v)) for t, v in up_res[0]["values"]]
            prev_up = None
            for ts, up_v in up_values:
                if prev_up is not None and prev_up < 1.0 and up_v >= 1.0:
                    restarts.append(ts)
                prev_up = up_v

    span_start = span_end = None
    prev = None
    ri = 0
    for ts, v in values:
        # apply target restarts that happened between prev sample and this one
        while ri < len(restarts) and restarts[ri] <= ts:
            span_start = span_end = None
            prev = None
            ri += 1
        if prev is None:
            prev = v
            if v > 0 and ts >= start_s:
                span_start = span_end = ts
            continue
        if v < prev:
            span_start = span_end = None
            if v > 0 and ts >= start_s:
                span_start = span_end = ts
        elif v > prev:
            if ts >= start_s:
                if span_start is None:
                    span_start = ts
                span_end = ts
        prev = v
    return (span_start, span_end) if span_start is not None else None


def bytes_by_layer(op_type: str, t: float | None = None) -> dict[str, int]:
    """Return {layer_digest: bytes} for stargz_fs_bytes_served{operation_type=op_type}
    aggregated by layer at instant t. Missing/unreachable prometheus → {}."""
    promql = (
        f'sum by (layer) (stargz_fs_bytes_served{{operation_type="{op_type}"}})'
    )
    out: dict[str, int] = {}
    for entry in query(promql, t):
        layer = entry.get("metric", {}).get("layer", "")
        val = entry.get("value", [None, "0"])[1]
        try:
            out[layer] = int(float(val))
        except (TypeError, ValueError):
            continue
    return out


def _get(path: str) -> list[dict]:
    try:
        with urlopen(f"{PROM_URL}{path}", timeout=HTTP_TIMEOUT_S) as r:
            return json.load(r).get("data", {}).get("result", [])
    except Exception as e:
        log.info(f"  prometheus query failed: {e}")
        return []
