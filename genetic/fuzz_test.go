package genetic_test

import (
	"math/rand"
	"testing"

	"github.com/Semplicementeio/Genetic-Algorithm/genetic"
)

func FuzzEngineConfig(f *testing.F) {
	// Add seed corpus entries
	f.Add(10, 5, 0.05, 0.8, 1, 1, int64(42))
	f.Add(50, 10, 0.10, 0.5, 2, 4, int64(100))
	f.Add(-1, 0, -0.5, 1.5, -2, 0, int64(0))

	f.Fuzz(func(t *testing.T, popSize, gens int, mutRate, crossRate float64, elitism, workers int, seed int64) {
		cfg := genetic.Config[[]byte]{
			PopulationSize: popSize,
			Generations:    gens,
			MutationRate:   mutRate,
			CrossoverRate:  crossRate,
			ElitismCount:   elitism,
			NumWorkers:     workers,
			Seed:           seed,
			Generator: func(rng *rand.Rand) []byte {
				return []byte{byte(rng.Intn(256)), byte(rng.Intn(256))}
			},
			FitnessFunc: func(g []byte) float64 {
				if len(g) < 2 {
					return 0.0
				}
				return float64(g[0]) + float64(g[1])
			},
			Selector:  genetic.NewTournamentSelection[[]byte](2),
			Crossover: genetic.NewSinglePointCrossover[[]byte, byte](),
			Mutator:   genetic.NewByteMutation(""),
		}

		err := cfg.Validate()
		if err != nil {
			// Engine creation should fail gracefully for invalid configs
			_, engineErr := genetic.NewEngine(cfg)
			if engineErr == nil {
				t.Fatalf("expected NewEngine to return error for invalid config: %v", cfg)
			}
			return
		}

		engine, err := genetic.NewEngine(cfg)
		if err != nil {
			t.Fatalf("unexpected error creating engine: %v", err)
		}

		// Keep fuzz executions cheap: only run bounded, valid configurations.
		if popSize > 200 || gens > 30 || workers > 16 {
			return
		}

		res, err := engine.Run()
		if err != nil {
			t.Fatalf("unexpected run error: %v", err)
		}

		// Invariants that must hold for every valid configuration.
		if len(res.FinalPopulation) != popSize {
			t.Fatalf("final population size %d, want %d", len(res.FinalPopulation), popSize)
		}
		if res.GenerationsEvaluated != len(res.History) || res.GenerationsEvaluated < 1 || res.GenerationsEvaluated > gens {
			t.Fatalf("inconsistent generation count %d (history %d, max %d)", res.GenerationsEvaluated, len(res.History), gens)
		}
		for _, ind := range res.FinalPopulation {
			if !ind.Evaluated {
				t.Fatal("unevaluated individual in final population")
			}
			if ind.Fitness > res.BestIndividual.Fitness {
				t.Fatalf("best individual (%f) is worse than a final individual (%f)", res.BestIndividual.Fitness, ind.Fitness)
			}
		}
		for i := 1; i < len(res.History); i++ {
			if res.History[i].Generation != res.History[i-1].Generation+1 {
				t.Fatal("history generations are not consecutive")
			}
		}
	})
}
