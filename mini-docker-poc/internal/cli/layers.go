package cli

import "mini-docker/internal/overlay"

// LayersDemo demonstrates Docker's image-layer stacking mechanism using
// OverlayFS, independent of the `run` command's container rootfs.
func LayersDemo(args []string) int {
	return overlay.RunDemo(args)
}
