package cli

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"syscall"
	"time"

	"mini-docker/internal/cgroups"
	"mini-docker/internal/container"
)

// Run creates the cgroup for a new container, then re-executes this same
// binary as `child` inside a fresh set of Linux namespaces (UTS, PID, mount,
// network, IPC) via /proc/self/exe. This mirrors how real container runtimes
// like runc bootstrap a container: the "namespace creation" step and the
// "namespace setup" step happen in two different process images, because
// some namespace effects (like a process seeing itself as PID 1) only take
// effect for a newly created child, not the process that calls unshare/clone
// itself.
func Run(args []string) int {
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	rootfsPath := fs.String("rootfs", "", "path to the container rootfs (required)")
	mem := fs.String("mem", "", "memory limit, e.g. 100m, 1g (empty = unlimited)")
	cpus := fs.Float64("cpus", 0, "CPU limit in cores, e.g. 0.5 (0 = unlimited)")
	pids := fs.Int("pids", 0, "max number of processes/threads (0 = unlimited)")
	hostname := fs.String("hostname", "mini-docker", "hostname to set inside the container")
	if err := fs.Parse(args); err != nil {
		return 1
	}
	command := fs.Args()

	if *rootfsPath == "" {
		fmt.Fprintln(os.Stderr, "run: --rootfs is required")
		return 1
	}
	if len(command) == 0 {
		fmt.Fprintln(os.Stderr, "run: a command is required, e.g.: mini-docker run --rootfs ./rootfs -- /bin/sh")
		return 1
	}
	memBytes, err := cgroups.ParseSize(*mem)
	if err != nil {
		fmt.Fprintln(os.Stderr, "run:", err)
		return 1
	}

	id := container.NewID()

	cg, err := cgroups.New(id, cgroups.Limits{
		MemoryBytes: memBytes,
		PidsMax:     *pids,
		CPUs:        *cpus,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "run: cgroup setup:", err)
		return 1
	}

	reexecArgs := append([]string{"child", "--rootfs", *rootfsPath, "--hostname", *hostname, "--"}, command...)
	cmd := exec.Command("/proc/self/exe", reexecArgs...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Cloneflags: syscall.CLONE_NEWUTS |
			syscall.CLONE_NEWPID |
			syscall.CLONE_NEWNS |
			syscall.CLONE_NEWNET |
			syscall.CLONE_NEWIPC,
	}

	if err := cmd.Start(); err != nil {
		fmt.Fprintln(os.Stderr, "run: start container:", err)
		_ = cg.Cleanup()
		return 1
	}

	// cmd.Process.Pid is the PID as seen from OUR (the parent's, i.e. host)
	// PID namespace -- exactly what cgroup.procs expects, even though the
	// child sees itself as PID 1 inside its own new PID namespace.
	if err := cg.AddProcess(cmd.Process.Pid); err != nil {
		fmt.Fprintln(os.Stderr, "run: add process to cgroup:", err)
	}

	_ = container.SaveState(container.State{
		ID:        id,
		Pid:       cmd.Process.Pid,
		Rootfs:    *rootfsPath,
		Command:   command,
		StartedAt: time.Now(),
	})

	waitErr := cmd.Wait()

	_ = container.RemoveState(id)
	if err := cg.Cleanup(); err != nil {
		fmt.Fprintln(os.Stderr, "run: cgroup cleanup:", err)
	}

	if waitErr != nil {
		if exitErr, ok := waitErr.(*exec.ExitError); ok {
			return exitErr.ExitCode()
		}
		fmt.Fprintln(os.Stderr, "run:", waitErr)
		return 1
	}
	return 0
}
