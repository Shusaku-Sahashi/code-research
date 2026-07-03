package cgroups

import (
	"fmt"
	"strconv"
	"strings"
)

// Limits holds the resource limits requested via CLI flags.
type Limits struct {
	MemoryBytes int64   // 0 means unlimited
	PidsMax     int     // 0 means unlimited
	CPUs        float64 // 0 means unlimited; 0.5 == half a CPU core
}

// ParseSize parses a human size like "100m", "1g", "512k" or a plain byte
// count into bytes. Returns 0 (unlimited) for an empty string.
func ParseSize(s string) (int64, error) {
	s = strings.TrimSpace(strings.ToLower(s))
	if s == "" {
		return 0, nil
	}

	mult := int64(1)
	switch {
	case strings.HasSuffix(s, "g"):
		mult = 1024 * 1024 * 1024
		s = strings.TrimSuffix(s, "g")
	case strings.HasSuffix(s, "m"):
		mult = 1024 * 1024
		s = strings.TrimSuffix(s, "m")
	case strings.HasSuffix(s, "k"):
		mult = 1024
		s = strings.TrimSuffix(s, "k")
	}

	n, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("invalid size %q: %w", s, err)
	}
	return n * mult, nil
}
