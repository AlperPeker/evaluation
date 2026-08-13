#!/bin/bash
set -euox pipefail

# -------------------------------------------------------------------
# Install HWE kernel (required for FUSE passthrough, needs 6.9+)
# and reboot. Run this before combined-node-setup.sh.
#
# Usage: sudo ./install-hwe-kernel.sh
# -------------------------------------------------------------------

if [[ $EUID -ne 0 ]]; then
  echo "Please run as root"
  exit 1
fi

KERNEL_MAJOR=$(uname -r | cut -d. -f1)
KERNEL_MINOR=$(uname -r | cut -d. -f2 | cut -d- -f1)
if [[ "$KERNEL_MAJOR" -gt 6 ]] || [[ "$KERNEL_MAJOR" -eq 6 && "$KERNEL_MINOR" -ge 9 ]]; then
  echo "Kernel $(uname -r) already >= 6.9 — nothing to do"
  exit 0
fi

apt-get update -q
apt-get install -y linux-generic-hwe-24.04

echo "HWE kernel installed. Rebooting..."
reboot
