package main

import (
	"os"
	"testing"

	"github.com/Semplicementeio/Genetic-Algorithm/examples/knapsack"
	"github.com/Semplicementeio/Genetic-Algorithm/examples/scheduling"
	"github.com/Semplicementeio/Genetic-Algorithm/examples/tsp"
)

// TestExamplesRunWithoutPanicking guards the runnable examples against API drift.
// (The optimization benchmark is excluded to keep `go test ./...` fast.)
func TestExamplesRunWithoutPanicking(t *testing.T) {
	// Silence example output unless the test fails.
	devNull, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer devNull.Close()
	orig := os.Stdout
	os.Stdout = devNull
	defer func() { os.Stdout = orig }()

	for _, seed := range []int64{1, 42} {
		knapsack.Run(seed)
		tsp.Run(seed)
		scheduling.Run(seed)
	}
}
