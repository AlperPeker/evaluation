#!/bin/bash
set -euox pipefail

sudo buildctl prune --all
rm -rf ~/.2dfs/blobs/* ~/.2dfs/uncompressed-keys/* ~/.2dfs/index/*
