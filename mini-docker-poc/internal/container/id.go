package container

import (
	"crypto/rand"
	"encoding/hex"
)

// NewID returns a random 12-hex-character container id, similar in spirit to
// Docker's short container IDs.
func NewID() string {
	buf := make([]byte, 6)
	if _, err := rand.Read(buf); err != nil {
		panic(err)
	}
	return hex.EncodeToString(buf)
}
