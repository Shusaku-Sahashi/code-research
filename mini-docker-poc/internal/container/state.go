package container

import (
	"encoding/json"
	"os"
	"path/filepath"
	"syscall"
	"time"
)

const stateDir = "/var/run/mini-docker/state"

// State is the on-disk record for a running container, used by `ps` to list
// containers. It is intentionally minimal: this is a learning tool, not a
// production container manager.
type State struct {
	ID        string    `json:"id"`
	Pid       int       `json:"pid"`
	Rootfs    string    `json:"rootfs"`
	Command   []string  `json:"command"`
	StartedAt time.Time `json:"started_at"`
}

func statePath(id string) string {
	return filepath.Join(stateDir, id+".json")
}

// SaveState writes the state file for a container. Called by `run` right
// after the child process has been started.
func SaveState(s State) error {
	if err := os.MkdirAll(stateDir, 0755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(statePath(s.ID), data, 0644)
}

// RemoveState deletes a container's state file. Called by `run` after the
// child process exits.
func RemoveState(id string) error {
	err := os.Remove(statePath(id))
	if os.IsNotExist(err) {
		return nil
	}
	return err
}

// ListStates returns all known container states, skipping stale entries whose
// process is no longer alive.
func ListStates() ([]State, error) {
	entries, err := os.ReadDir(stateDir)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}

	var states []State
	for _, e := range entries {
		if e.IsDir() || filepath.Ext(e.Name()) != ".json" {
			continue
		}
		data, err := os.ReadFile(filepath.Join(stateDir, e.Name()))
		if err != nil {
			continue
		}
		var s State
		if err := json.Unmarshal(data, &s); err != nil {
			continue
		}
		if processAlive(s.Pid) {
			states = append(states, s)
		}
	}
	return states, nil
}

func processAlive(pid int) bool {
	// On Linux, os.FindProcess always succeeds; signal 0 is the standard way
	// to probe whether a pid exists without actually sending a signal.
	proc, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	return proc.Signal(syscall.Signal(0)) == nil
}
