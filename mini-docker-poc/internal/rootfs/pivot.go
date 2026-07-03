package rootfs

import (
	"fmt"
	"os"
	"path/filepath"

	"golang.org/x/sys/unix"
)

// PivotInto replaces the calling process's root filesystem with newRoot using
// pivot_root(2), the same primitive real container runtimes (runc, dockerd's
// containerd-shim) use. Must be called from inside a process that already has
// its own mount namespace (CLONE_NEWNS), otherwise the pivot would affect the
// host.
func PivotInto(newRoot string) error {
	absRoot, err := filepath.Abs(newRoot)
	if err != nil {
		return fmt.Errorf("resolve rootfs path: %w", err)
	}

	// Make our mount namespace's tree private+recursive so mount/unmount
	// events here never propagate to the host (and vice versa). Required by
	// pivot_root on hosts where the root mount is "shared"; a no-op on hosts
	// where it is already "private".
	if err := unix.Mount("", "/", "", unix.MS_PRIVATE|unix.MS_REC, ""); err != nil {
		return fmt.Errorf("remount / private: %w", err)
	}

	// pivot_root requires new_root to be a mount point, so bind-mount it onto
	// itself first.
	if err := unix.Mount(absRoot, absRoot, "", unix.MS_BIND|unix.MS_REC, ""); err != nil {
		return fmt.Errorf("bind mount rootfs onto itself: %w", err)
	}

	oldRoot := filepath.Join(absRoot, ".old_root")
	if err := os.MkdirAll(oldRoot, 0700); err != nil {
		return fmt.Errorf("create old_root mountpoint: %w", err)
	}

	if err := unix.PivotRoot(absRoot, oldRoot); err != nil {
		return fmt.Errorf("pivot_root: %w", err)
	}

	if err := os.Chdir("/"); err != nil {
		return fmt.Errorf("chdir to new root: %w", err)
	}

	// The old root is now mounted at /.old_root inside the new root; detach it
	// so the container can no longer see the host filesystem at all.
	if err := unix.Unmount("/.old_root", unix.MNT_DETACH); err != nil {
		return fmt.Errorf("unmount old root: %w", err)
	}
	// Best-effort cleanup of the now-empty mountpoint directory.
	_ = os.RemoveAll("/.old_root")

	return nil
}

// MountProc mounts a fresh procfs at /proc, reflecting the container's own
// PID namespace rather than the host's.
func MountProc() error {
	if err := os.MkdirAll("/proc", 0755); err != nil {
		return err
	}
	return unix.Mount("proc", "/proc", "proc", 0, "")
}
