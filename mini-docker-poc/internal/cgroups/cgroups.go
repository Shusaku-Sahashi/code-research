package cgroups

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
)

const (
	cgroupRoot  = "/sys/fs/cgroup"
	groupName   = "mini-docker"
	cpuPeriodUs = 100000
)

var controllers = []string{"memory", "pids", "cpu"}

// CGroup represents the set of cgroup v1 controller directories created for a
// single container.
type CGroup struct {
	ID    string
	paths map[string]string
}

// New creates a mini-docker/<id> cgroup under each v1 controller this tool
// uses (memory, pids, cpu) and writes the requested limits. Real Docker
// abstracts this behind a pluggable "cgroup driver"; here it's hardcoded to
// cgroup v1 paths/filenames since that's what this host provides.
func New(id string, l Limits) (*CGroup, error) {
	for _, c := range controllers {
		if _, err := os.Stat(filepath.Join(cgroupRoot, c)); err != nil {
			return nil, fmt.Errorf("cgroup v1 controller %q not available at %s (this tool targets cgroup v1; a cgroup v2-only host needs different file names): %w", c, cgroupRoot, err)
		}
	}

	cg := &CGroup{ID: id, paths: map[string]string{}}
	for _, c := range controllers {
		p := filepath.Join(cgroupRoot, c, groupName, id)
		if err := os.MkdirAll(p, 0755); err != nil {
			return nil, fmt.Errorf("create cgroup dir %s: %w", p, err)
		}
		cg.paths[c] = p
	}

	if l.MemoryBytes > 0 {
		if err := writeFile(cg.paths["memory"], "memory.limit_in_bytes", itoa(l.MemoryBytes)); err != nil {
			return nil, err
		}
		// Swap accounting file is absent on kernels without swapaccount=1;
		// only set it if present.
		if fileExists(cg.paths["memory"], "memory.memsw.limit_in_bytes") {
			_ = writeFile(cg.paths["memory"], "memory.memsw.limit_in_bytes", itoa(l.MemoryBytes))
		}
	}

	if l.PidsMax > 0 {
		if err := writeFile(cg.paths["pids"], "pids.max", strconv.Itoa(l.PidsMax)); err != nil {
			return nil, err
		}
	}

	if l.CPUs > 0 {
		quota := int64(l.CPUs * cpuPeriodUs)
		if err := writeFile(cg.paths["cpu"], "cpu.cfs_period_us", itoa(cpuPeriodUs)); err != nil {
			return nil, err
		}
		if err := writeFile(cg.paths["cpu"], "cpu.cfs_quota_us", itoa(quota)); err != nil {
			return nil, err
		}
	}

	return cg, nil
}

// AddProcess adds pid (as seen from the caller's own, i.e. host, PID
// namespace) to every controller this CGroup manages.
func (cg *CGroup) AddProcess(pid int) error {
	for _, p := range cg.paths {
		if err := writeFile(p, "cgroup.procs", strconv.Itoa(pid)); err != nil {
			return fmt.Errorf("add pid %d to %s: %w", pid, p, err)
		}
	}
	return nil
}

// Cleanup removes the cgroup directories. Must only be called after the
// tracked process has fully exited (e.g. after cmd.Wait() returns); the
// kernel refuses rmdir on a cgroup that still has member processes (EBUSY).
func (cg *CGroup) Cleanup() error {
	var firstErr error
	for _, p := range cg.paths {
		if err := os.Remove(p); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

func writeFile(dir, name, value string) error {
	return os.WriteFile(filepath.Join(dir, name), []byte(value), 0644)
}

func fileExists(dir, name string) bool {
	_, err := os.Stat(filepath.Join(dir, name))
	return err == nil
}

func itoa(n int64) string {
	return strconv.FormatInt(n, 10)
}
