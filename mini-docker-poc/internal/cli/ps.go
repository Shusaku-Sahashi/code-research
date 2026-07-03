package cli

import (
	"fmt"
	"os"
	"strings"
	"text/tabwriter"
	"time"

	"mini-docker/internal/container"
)

// Ps lists currently running mini-docker containers, similar to `docker ps`.
func Ps(args []string) int {
	states, err := container.ListStates()
	if err != nil {
		fmt.Fprintln(os.Stderr, "mini-docker ps:", err)
		return 1
	}

	w := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "CONTAINER ID\tPID\tCOMMAND\tROOTFS\tUPTIME")
	for _, s := range states {
		fmt.Fprintf(w, "%s\t%d\t%s\t%s\t%s\n",
			s.ID, s.Pid, strings.Join(s.Command, " "), s.Rootfs, time.Since(s.StartedAt).Round(time.Second))
	}
	w.Flush()
	return 0
}
