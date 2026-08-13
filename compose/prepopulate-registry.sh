#!/usr/bin/env sh
set -eu

# containerd setup
echo "==> Starting containerd"
/usr/local/bin/containerd &

# wait for containerd socket
until ctr version >/dev/null 2>&1; do
  sleep 0.2
done

# prepopulate registry
REGISTRY="registry:5000"
echo "==> Waiting for registry..."
until curl -fsS "http://${REGISTRY}/v2/" >/dev/null; do
  sleep 0.2
done

REPO="bert-split"
STANDARD_SRC="ghcr.io/bohdangarchu/bert-split:org"
STARGZ_SRC="ghcr.io/bohdangarchu/bert-split:esgz"
STANDARD_DST="$REGISTRY/$REPO:org"
STARGZ_DST="$REGISTRY/$REPO:esgz"

quiet_run() {
  if ! output="$("$@" 2>&1)"; then
    echo "FAILED: $*" >&2
    echo "$output" >&2
    exit 1
  fi
}

check_manifest() {
  repo="$1"
  tag="$2"
  curl -fsS -o /dev/null \
    -H 'Accept: application/vnd.oci.image.manifest.v1+json' \
    "http://${REGISTRY}/v2/${repo}/manifests/${tag}"
}

# early exit if images aleady exist
if check_manifest "$REPO" "org" && check_manifest "$REPO" "esgz"; then
  echo "==> Images already present in registry, skipping pull/push"
  exit 0
fi

echo "==> Pulling STANDARD image"
quiet_run ctr image pull "$STANDARD_SRC"
quiet_run ctr image tag "$STANDARD_SRC" "$STANDARD_DST"
quiet_run ctr image push --plain-http "$STANDARD_DST"
quiet_run ctr image rm "$STANDARD_SRC" "$STANDARD_DST"

echo "==> Pulling STARGZ image"
quiet_run ctr-remote image pull "$STARGZ_SRC"
quiet_run ctr image tag "$STARGZ_SRC" "$STARGZ_DST"
quiet_run ctr image push --plain-http "$STARGZ_DST"
quiet_run ctr image rm "$STARGZ_SRC" "$STARGZ_DST"

echo "==> DONE: registry populated"
