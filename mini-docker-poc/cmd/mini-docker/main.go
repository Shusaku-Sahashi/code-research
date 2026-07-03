package main

import (
	"fmt"
	"os"

	"mini-docker/internal/cli"
)

func usage() {
	fmt.Fprintln(os.Stderr, `mini-docker: a small educational container runtime

Usage:
  mini-docker run --rootfs <path> [--mem 100m] [--cpus 0.5] [--pids 64] [--hostname name] -- <cmd> [args...]
  mini-docker ps
  mini-docker layers-demo [--dir ./overlay-demo]`)
}

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}

	var code int
	switch os.Args[1] {
	case "run":
		code = cli.Run(os.Args[2:])
	case "child":
		// Internal reexec target invoked by `run` via /proc/self/exe. Not meant
		// to be invoked directly by end users.
		code = cli.Child(os.Args[2:])
	case "ps":
		code = cli.Ps(os.Args[2:])
	case "layers-demo":
		code = cli.LayersDemo(os.Args[2:])
	default:
		usage()
		code = 1
	}
	os.Exit(code)
}
