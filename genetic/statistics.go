package genetic

import (
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"time"
)

// GenerationStats records key metrics for a single generation.
type GenerationStats struct {
	Generation   int           `json:"generation"`
	BestFitness  float64       `json:"best_fitness"`
	AvgFitness   float64       `json:"avg_fitness"`
	WorstFitness float64       `json:"worst_fitness"`
	StdDev       float64       `json:"std_dev"`
	Duration     time.Duration `json:"duration_ns"`
}

// History stores generational statistics over an entire run.
type History []GenerationStats

// Record GenerationStats into history.
func (h *History) Record(stats GenerationStats) {
	*h = append(*h, stats)
}

// ExportCSV writes generational metrics as CSV data.
func (h History) ExportCSV(w io.Writer) error {
	writer := csv.NewWriter(w)
	defer writer.Flush()

	// Write header
	header := []string{"generation", "best_fitness", "avg_fitness", "worst_fitness", "std_dev", "duration_ms"}
	if err := writer.Write(header); err != nil {
		return fmt.Errorf("failed to write CSV header: %w", err)
	}

	for _, s := range h {
		row := []string{
			fmt.Sprintf("%d", s.Generation),
			fmt.Sprintf("%.6f", s.BestFitness),
			fmt.Sprintf("%.6f", s.AvgFitness),
			fmt.Sprintf("%.6f", s.WorstFitness),
			fmt.Sprintf("%.6f", s.StdDev),
			fmt.Sprintf("%.3f", float64(s.Duration.Microseconds())/1000.0),
		}
		if err := writer.Write(row); err != nil {
			return fmt.Errorf("failed to write CSV row: %w", err)
		}
	}

	return nil
}

// CalculateStdDev computes the standard deviation of fitness in a population.
func CalculateStdDev[T any](pop Population[T], avg float64) float64 {
	if len(pop) <= 1 {
		return 0.0
	}
	sumSqDiff := 0.0
	for _, ind := range pop {
		diff := ind.Fitness - avg
		sumSqDiff += diff * diff
	}
	return math.Sqrt(sumSqDiff / float64(len(pop)))
}
