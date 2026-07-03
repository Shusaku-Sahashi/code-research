package overlay

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"golang.org/x/sys/unix"
)

// RunDemo builds two sample "image layers", mounts them with the real lower
// one on top via OverlayFS, and narrates what happened. This is the same
// mechanism Docker's overlay2 graph driver uses to stack image layers plus one
// writable container layer into a single merged filesystem view.
func RunDemo(args []string) int {
	fs := flag.NewFlagSet("layers-demo", flag.ExitOnError)
	dir := fs.String("dir", "./overlay-demo", "working directory for the demo")
	if err := fs.Parse(args); err != nil {
		return 1
	}

	layer1 := filepath.Join(*dir, "layer1-lower") // e.g. "base image" layer
	layer2 := filepath.Join(*dir, "layer2-lower") // e.g. "app" layer, applied on top
	upper := filepath.Join(*dir, "upper")         // writable "container layer"
	work := filepath.Join(*dir, "work")           // overlayfs internal bookkeeping dir
	merged := filepath.Join(*dir, "merged")       // the combined view a container sees

	if err := writeSample(layer1, "hello.txt", "from layer1 (base image)\n"); err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	if err := writeSample(layer1, "shared.txt", "layer1 version\n"); err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	if err := writeSample(layer2, "shared.txt", "layer2 version (overwrites layer1)\n"); err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	if err := writeSample(layer2, "app.txt", "from layer2 (app layer)\n"); err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	for _, d := range []string{upper, work, merged} {
		if err := os.MkdirAll(d, 0755); err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 1
		}
	}

	// overlayfs lowerdir list is colon-separated, LEFT-most = highest
	// priority. layer2 (the "later"/"app" layer in Docker terms) must come
	// first so it wins over layer1 (the "base image" layer) for shared.txt.
	opts := fmt.Sprintf("lowerdir=%s:%s,upperdir=%s,workdir=%s", abs(layer2), abs(layer1), abs(upper), abs(work))
	if err := unix.Mount("overlay", merged, "overlay", 0, opts); err != nil {
		fmt.Fprintf(os.Stderr, "mount overlay: %v\n", err)
		return 1
	}

	fmt.Println("Mounted overlay filesystem at:", merged)
	fmt.Println()
	fmt.Println("Layer stack (highest priority first): layer2-lower, layer1-lower, + upper (writable)")
	fmt.Println()
	fmt.Println("Try in another shell:")
	fmt.Printf("  cat %s        # -> layer1 (base) only file\n", filepath.Join(merged, "hello.txt"))
	fmt.Printf("  cat %s       # -> layer2 wins over layer1\n", filepath.Join(merged, "shared.txt"))
	fmt.Printf("  cat %s          # -> layer2 only file\n", filepath.Join(merged, "app.txt"))
	fmt.Printf("  echo hi > %s/new.txt && ls %s   # -> new.txt copy-up'd into the writable layer\n", merged, upper)
	fmt.Println()
	fmt.Println("Press Enter to unmount and clean up...")
	bufio.NewReader(os.Stdin).ReadString('\n')

	if err := unix.Unmount(merged, 0); err != nil {
		fmt.Fprintf(os.Stderr, "unmount overlay: %v\n", err)
		return 1
	}
	if err := os.RemoveAll(*dir); err != nil {
		fmt.Fprintf(os.Stderr, "cleanup %s: %v\n", *dir, err)
		return 1
	}
	return 0
}

func writeSample(dir, name, content string) error {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, name), []byte(content), 0644)
}

func abs(p string) string {
	a, err := filepath.Abs(p)
	if err != nil {
		return p
	}
	return a
}
