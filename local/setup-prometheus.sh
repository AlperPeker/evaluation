#!/usr/bin/env bash
set -euo pipefail

PROMETHEUS_VERSION=3.9.1
NODE_EXPORTER_VERSION=1.9.1
ARCH=amd64

if command -v prometheus &>/dev/null; then
    echo "prometheus already installed: $(prometheus --version 2>&1 | head -1)"
else
    echo "installing prometheus v${PROMETHEUS_VERSION}..."
    TMP=$(mktemp -d)
    curl -sL "https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/prometheus-${PROMETHEUS_VERSION}.linux-${ARCH}.tar.gz" \
        | tar -xz -C "$TMP" \
            "prometheus-${PROMETHEUS_VERSION}.linux-${ARCH}/prometheus" \
            "prometheus-${PROMETHEUS_VERSION}.linux-${ARCH}/promtool"
    sudo mv "$TMP/prometheus-${PROMETHEUS_VERSION}.linux-${ARCH}/prometheus" /usr/local/bin/prometheus
    sudo mv "$TMP/prometheus-${PROMETHEUS_VERSION}.linux-${ARCH}/promtool" /usr/local/bin/promtool
    rm -rf "$TMP"
    echo "prometheus $(prometheus --version 2>&1 | head -1) installed"
fi

if command -v node_exporter &>/dev/null; then
    echo "node_exporter already installed: $(node_exporter --version 2>&1 | head -1)"
else
    echo "installing node_exporter v${NODE_EXPORTER_VERSION}..."
    TMP=$(mktemp -d)
    curl -sL "https://github.com/prometheus/node_exporter/releases/download/v${NODE_EXPORTER_VERSION}/node_exporter-${NODE_EXPORTER_VERSION}.linux-${ARCH}.tar.gz" \
        | tar -xz -C "$TMP" \
            "node_exporter-${NODE_EXPORTER_VERSION}.linux-${ARCH}/node_exporter"
    sudo mv "$TMP/node_exporter-${NODE_EXPORTER_VERSION}.linux-${ARCH}/node_exporter" /usr/local/bin/node_exporter
    rm -rf "$TMP"
    echo "node_exporter $(node_exporter --version 2>&1 | head -1) installed"
fi
