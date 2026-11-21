package benchmarks

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"testing"

	"github.com/Semplicementeio/Genetic-Algorithm/genetic"
)

// Heavy fitness calculation simulating realistic optimization problem (e.g. Ackley function)
func ackleyFitness(g []float64) float64 {
	d := float64(len(g))
	sum1 := 0.0
	sum2 := 0.0
	for _, x := range g {
		sum1 += x * x
		sum2 += math.Cos(2 * math.Pi * x)
	}
	term1 := -20 * math.Exp(-0.2*math.Sqrt(sum1/d))
	term2 := -math.Exp(sum2 / d)
	val := term1 + term2 + 20 + math.E

	// Return negative Ackley value so that maximizing fitness minimizes Ackley distance to 0
	return -val
}

func BenchmarkPopulationSizes(b *testing.B) {
	popSizes := []int{100, 500, 1000, 5000}

	for _, popSize := range popSizes {
		b.Run(fmt.Sprintf("PopSize-%d", popSize), func(b *testing.B) {
			cfg := genetic.Config[[]float64]{
				PopulationSize: popSize,
				Generations:    50,
				MutationRate:   0.05,
				CrossoverRate:  0.8,
				ElitismCount:   2,
				NumWorkers:     4,
				Seed:           42,
				Generator: func(rng *rand.Rand) []float64 {
					g := make([]float64, 10)
					for i := range g {
						g[i] = rng.Float64()*10.0 - 5.0
					}
					return g
				},
				FitnessFunc: ackleyFitness,
				Selector:    genetic.NewTournamentSelection[[]float64](3),
				Crossover:   genetic.NewUniformCrossover[[]float64, float64](0.5),
				Mutator:     genetic.NewGaussianMutation(0.2),
			}

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				engine, err := genetic.NewEngine(cfg)
				if err != nil {
					b.Fatalf("failed to create engine: %v", err)
				}
				_, _ = engine.Run()
			}
		})
	}
}

// BenchmarkWorkerPoolScalability measures how the engine scales with NumWorkers for
// two fitness costs on the same 50-dimensional problem:
//
//   - cheap:    one Ackley evaluation (~0.5 µs). The run is dominated by the engine's own
//     work (breeding, statistics, memory allocation), so this shows the framework
//     overhead and the ceiling for very light fitness functions.
//   - moderate: 50 Ackley evaluations (~28 µs). Fitness dominates, which is the regime
//     where parallel evaluation is meant to pay off.
//
// Run it on an otherwise idle machine; background load makes the results meaningless.
func BenchmarkWorkerPoolScalability(b *testing.B) {
	workloads := []struct {
		name string
		reps int
	}{
		{"cheap", 1},
		{"moderate", 50},
	}
	workersList := []int{1, 2, 4, 6, 8, 12}

	for _, wl := range workloads {
		for _, workers := range workersList {
			b.Run(fmt.Sprintf("%s/Workers-%d", wl.name, workers), func(b *testing.B) {
				reps := wl.reps
				cfg := genetic.Config[[]float64]{
					PopulationSize: 1000,
					Generations:    20,
					MutationRate:   0.05,
					CrossoverRate:  0.8,
					ElitismCount:   2,
					NumWorkers:     workers,
					Seed:           42,
					Generator: func(rng *rand.Rand) []float64 {
						g := make([]float64, 50)
						for i := range g {
							g[i] = rng.Float64()*10.0 - 5.0
						}
						return g
					},
					FitnessFunc: func(g []float64) float64 {
						sum := 0.0
						for i := 0; i < reps; i++ {
							sum += ackleyFitness(g)
						}
						return sum
					},
					Selector:  genetic.NewTournamentSelection[[]float64](3),
					Crossover: genetic.NewUniformCrossover[[]float64, float64](0.5),
					Mutator:   genetic.NewGaussianMutation(0.2),
				}

				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					engine, err := genetic.NewEngine(cfg)
					if err != nil {
						b.Fatalf("failed to create engine: %v", err)
					}
					if _, err := engine.Run(); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

// BenchmarkHardwareCeiling runs the same fitness work as the "moderate" workload of
// BenchmarkWorkerPoolScalability (19,962 evaluations of 50 x Ackley-50D) on bare
// goroutines, with no genetic-algorithm code at all. It is the best speedup any
// implementation could reach on this machine (core count, SMT, clock boost, memory
// bandwidth), so engine results should be read against it, not against N x workers.
func BenchmarkHardwareCeiling(b *testing.B) {
	const evaluations = 19962
	g := make([]float64, 50)
	for i := range g {
		g[i] = float64(i)*0.2 - 5
	}

	for _, workers := range []int{1, 2, 4, 6, 8, 12} {
		b.Run(fmt.Sprintf("Workers-%d", workers), func(b *testing.B) {
			sinks := make([]float64, workers*16) // padded to avoid false sharing
			for i := 0; i < b.N; i++ {
				var wg sync.WaitGroup
				wg.Add(workers)
				for w := 0; w < workers; w++ {
					go func(w int) {
						defer wg.Done()
						sum := 0.0
						for n := 0; n < evaluations/workers; n++ {
							for r := 0; r < 50; r++ {
								sum += ackleyFitness(g)
							}
						}
						sinks[w*16] = sum
					}(w)
				}
				wg.Wait()
			}
		})
	}
}

func BenchmarkIslandModelScalability(b *testing.B) {
	islandsList := []int{1, 2, 4, 8, 16}

	for _, islands := range islandsList {
		b.Run(fmt.Sprintf("Islands-%d", islands), func(b *testing.B) {
			baseCfg := genetic.Config[[]float64]{
				PopulationSize: 100,
				Generations:    20,
				MutationRate:   0.05,
				CrossoverRate:  0.8,
				ElitismCount:   2,
				NumWorkers:     2,
				Seed:           42,
				Generator: func(rng *rand.Rand) []float64 {
					g := make([]float64, 10)
					for i := range g {
						g[i] = rng.Float64()*10.0 - 5.0
					}
					return g
				},
				FitnessFunc: ackleyFitness,
				Selector:    genetic.NewTournamentSelection[[]float64](3),
				Crossover:   genetic.NewUniformCrossover[[]float64, float64](0.5),
				Mutator:     genetic.NewGaussianMutation(0.2),
			}

			if islands == 1 {
				b.ResetTimer()
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					eng, _ := genetic.NewEngine(baseCfg)
					_, _ = eng.Run()
				}
				return
			}

			islandCfg := genetic.IslandConfig[[]float64]{
				NumIslands:           islands,
				IslandPopulationSize: 50,
				EpochGenerations:     5,
				MigrationCount:       2,
				TotalGenerations:     20,
				Topology:             genetic.RingTopology,
				BaseConfig:           baseCfg,
			}

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				islandEng, err := genetic.NewIslandEngine(islandCfg)
				if err != nil {
					b.Fatalf("failed to create island engine: %v", err)
				}
				_, _ = islandEng.Run()
			}
		})
	}
}
