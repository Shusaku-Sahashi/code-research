#!/usr/bin/env bash
# Builds a tiny container rootfs backed by a single busybox-static binary.
# No registry pull, no debootstrap -- just enough of a userland to explore
# namespace/cgroup isolation with `mini-docker run`.
set -euo pipefail

ROOTFS_DIR="${1:-./rootfs}"

if ! command -v busybox >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y busybox-static
fi

mkdir -p "$ROOTFS_DIR"/{bin,proc,sys,dev,etc,tmp,root}
cp "$(command -v busybox)" "$ROOTFS_DIR/bin/busybox"

for cmd in sh ls cat echo ps mount umount mkdir rm cp mv touch grep sleep \
           hostname top ping free df du chmod chown ln kill sed awk id whoami env; do
  ln -sf busybox "$ROOTFS_DIR/bin/$cmd"
done

echo "root:x:0:0:root:/root:/bin/sh" > "$ROOTFS_DIR/etc/passwd"
echo "root:x:0:" > "$ROOTFS_DIR/etc/group"

mknod -m 666 "$ROOTFS_DIR/dev/null" c 1 3 2>/dev/null || true
mknod -m 666 "$ROOTFS_DIR/dev/zero" c 1 5 2>/dev/null || true
mknod -m 666 "$ROOTFS_DIR/dev/random" c 1 8 2>/dev/null || true
mknod -m 666 "$ROOTFS_DIR/dev/urandom" c 1 9 2>/dev/null || true

echo "rootfs prepared at $ROOTFS_DIR"
