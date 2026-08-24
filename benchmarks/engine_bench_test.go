package benchmarks

import (
	"fmt"
	"math"
	"math/rand"
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

func BenchmarkWorkerPoolScalability(b *testing.B) {
	workersList := []int{1, 2, 4, 8, 16}

	for _, workers := range workersList {
		b.Run(fmt.Sprintf("Workers-%d", workers), func(b *testing.B) {
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
					// Add synthetic CPU work to fitness function
					sum := 0.0
					for i := 0; i < 500; i++ {
						sum += ackleyFitness(g)
					}
					return sum
				},
				Selector:  genetic.NewTournamentSelection[[]float64](3),
				Crossover: genetic.NewUniformCrossover[[]float64, float64](0.5),
				Mutator:   genetic.NewGaussianMutation(0.2),
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
