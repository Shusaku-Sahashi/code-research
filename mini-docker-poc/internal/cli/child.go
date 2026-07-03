package cli

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"syscall"

	"golang.org/x/sys/unix"

	"mini-docker/internal/rootfs"
)

// Child is the hidden reexec target that `run` launches via
// /proc/self/exe with new namespaces already applied (via
// SysProcAttr.Cloneflags on the parent's exec.Command). It sets the
// hostname, pivots into the container rootfs, mounts a fresh /proc, and
// finally execs the requested command so it becomes PID 1 inside the
// container.
func Child(args []string) int {
	fs := flag.NewFlagSet("child", flag.ExitOnError)
	rootfsPath := fs.String("rootfs", "", "path to the container rootfs")
	hostname := fs.String("hostname", "mini-docker", "hostname to set inside the container")
	if err := fs.Parse(args); err != nil {
		return 1
	}
	command := fs.Args()
	if *rootfsPath == "" || len(command) == 0 {
		fmt.Fprintln(os.Stderr, "child: --rootfs and a command are required")
		return 1
	}

	if err := syscall.Sethostname([]byte(*hostname)); err != nil {
		fmt.Fprintln(os.Stderr, "child: sethostname:", err)
		return 1
	}

	if err := rootfs.PivotInto(*rootfsPath); err != nil {
		fmt.Fprintln(os.Stderr, "child: pivot root:", err)
		return 1
	}

	if err := rootfs.MountProc(); err != nil {
		fmt.Fprintln(os.Stderr, "child: mount /proc:", err)
		return 1
	}

	binPath, err := exec.LookPath(command[0])
	if err != nil {
		fmt.Fprintf(os.Stderr, "child: %s: %v\n", command[0], err)
		return 1
	}

	// Exec replaces this process's image entirely; the target command
	// becomes PID 1 inside the container's PID namespace. This is why
	// mini-docker has no init/zombie-reaping logic: it inherits Docker's
	// historical "PID 1 problem" by design, as a single-process-per-container
	// learning tool.
	env := os.Environ()
	if err := unix.Exec(binPath, command, env); err != nil {
		fmt.Fprintf(os.Stderr, "child: exec %s: %v\n", binPath, err)
		return 1
	}
	return 0 // unreachable
}
