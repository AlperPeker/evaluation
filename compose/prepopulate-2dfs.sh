#!/usr/bin/env sh
set -eu

# containerd setup
echo "==> Starting containerd"
/usr/local/bin/containerd &

# wait for containerd socket
until ctr version >/dev/null 2>&1; do
  sleep 0.2
done

cd /2dfs
tdfs build ghcr.io/stargz-containers/python:3.7-esgz registry:5000/python:3.7-esgz
tdfs image push --force-http registry:5000/library/python:3.7-esgz
