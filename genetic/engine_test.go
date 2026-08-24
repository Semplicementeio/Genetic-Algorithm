package genetic_test

import (
	"bytes"
	"errors"
	"math/rand"
	"testing"

	"github.com/Semplicementeio/Genetic-Algorithm/genetic"
)

// Helper: Target string evolution setup
func targetStringSetup(target string) genetic.Config[[]byte] {
	alphabet := "abcdefghijklmnopqrstuvwxyz "
	targetBytes := []byte(target)

	return genetic.Config[[]byte]{
		PopulationSize: 100,
		Generations:    100,
		MutationRate:   0.05,
		CrossoverRate:  0.8,
		ElitismCount:   2,
		NumWorkers:     2,
		Seed:           42,
		Generator: func(rng *rand.Rand) []byte {
			res := make([]byte, len(targetBytes))
			for i := range res {
				res[i] = alphabet[rng.Intn(len(alphabet))]
			}
			return res
		},
		FitnessFunc: func(g []byte) float64 {
			fit := 0.0
			for i := range g {
				if g[i] == targetBytes[i] {
					fit += 1.0
				}
			}
			return fit
		},
		Selector:  genetic.NewTournamentSelection[[]byte](3),
		Crossover: genetic.NewSinglePointCrossover[[]byte, byte](),
		Mutator:   genetic.NewByteMutation(alphabet),
		GenomeCopier: func(g []byte) []byte {
			c := make([]byte, len(g))
			copy(c, g)
			return c
		},
	}
}

func TestConfigValidation(t *testing.T) {
	t.Run("Valid Config", func(t *testing.T) {
		cfg := targetStringSetup("hello")
		if err := cfg.Validate(); err != nil {
			t.Fatalf("expected valid config, got: %v", err)
		}
	})

	t.Run("Invalid Population Size", func(t *testing.T) {
		cfg := targetStringSetup("hello")
		cfg.PopulationSize = 1
		if err := cfg.Validate(); !errors.Is(err, genetic.ErrInvalidPopulationSize) {
			t.Fatalf("expected ErrInvalidPopulationSize, got %v", err)
		}
	})

	t.Run("Invalid Mutation Rate", func(t *testing.T) {
		cfg := targetStringSetup("hello")
		cfg.MutationRate = 1.5
		if err := cfg.Validate(); !errors.Is(err, genetic.ErrInvalidMutationRate) {
			t.Fatalf("expected ErrInvalidMutationRate, got %v", err)
		}
	})

	t.Run("Nil Fitness Function", func(t *testing.T) {
		cfg := targetStringSetup("hello")
		cfg.FitnessFunc = nil
		if err := cfg.Validate(); !errors.Is(err, genetic.ErrNilFitnessFunc) {
			t.Fatalf("expected ErrNilFitnessFunc, got %v", err)
		}
	})
}

func TestReproducibility(t *testing.T) {
	cfg1 := targetStringSetup("reproducible string test")
	cfg1.Seed = 12345

	cfg2 := targetStringSetup("reproducible string test")
	cfg2.Seed = 12345

	eng1, err := genetic.NewEngine(cfg1)
	if err != nil {
		t.Fatalf("failed to create engine 1: %v", err)
	}

	eng2, err := genetic.NewEngine(cfg2)
	if err != nil {
		t.Fatalf("failed to create engine 2: %v", err)
	}

	res1, err := eng1.Run()
	if err != nil {
		t.Fatalf("run 1 failed: %v", err)
	}

	res2, err := eng2.Run()
	if err != nil {
		t.Fatalf("run 2 failed: %v", err)
	}

	if !bytes.Equal(res1.BestIndividual.Genome, res2.BestIndividual.Genome) {
		t.Fatalf("reproducibility failed: genome mismatch '%s' vs '%s'", string(res1.BestIndividual.Genome), string(res2.BestIndividual.Genome))
	}

	if res1.BestIndividual.Fitness != res2.BestIndividual.Fitness {
		t.Fatalf("reproducibility failed: fitness mismatch %f vs %f", res1.BestIndividual.Fitness, res2.BestIndividual.Fitness)
	}

	if len(res1.History) != len(res2.History) {
		t.Fatalf("history length mismatch: %d vs %d", len(res1.History), len(res2.History))
	}

	for i := range res1.History {
		if res1.History[i].BestFitness != res2.History[i].BestFitness {
			t.Errorf("generation %d fitness mismatch: %f vs %f", i, res1.History[i].BestFitness, res2.History[i].BestFitness)
		}
	}
}

func TestPropertySelectionBias(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	pop := genetic.Population[int]{
		{Genome: 1, Fitness: 1.0},
		{Genome: 2, Fitness: 2.0},
		{Genome: 3, Fitness: 100.0}, // Dominant individual
		{Genome: 4, Fitness: 3.0},
	}

	ts := genetic.NewTournamentSelection[int](3)
	selectedDominantCount := 0
	trials := 1000

	for i := 0; i < trials; i++ {
		ind := ts.Select(pop, rng)
		if ind.Genome == 3 {
			selectedDominantCount++
		}
	}

	// In a population of 4 with K=3 tournament, probability of picking the dominant item is 1 - (3/4)^3 = 57.81%
	// Uniform random baseline is 25%.
	if selectedDominantCount < 500 {
		t.Fatalf("expected dominant individual to be selected >50%% of time, got %d/%d", selectedDominantCount, trials)
	}
}

func TestPropertyOrderCrossoverPermutation(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	p1 := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	p2 := []int{9, 8, 7, 6, 5, 4, 3, 2, 1, 0}

	ox := genetic.NewOrderCrossover[[]int, int]()
	c1, c2 := ox.Cross(p1, p2, rng)

	verifyPermutation := func(child []int) {
		if len(child) != len(p1) {
			t.Fatalf("child length mismatch: got %d, expected %d", len(child), len(p1))
		}
		seen := make(map[int]bool)
		for _, val := range child {
			if seen[val] {
				t.Fatalf("duplicate value %d in permutation child: %v", val, child)
			}
			seen[val] = true
		}
	}

	verifyPermutation(c1)
	verifyPermutation(c2)
}

func TestCSVExport(t *testing.T) {
	cfg := targetStringSetup("test csv export")
	cfg.Generations = 5
	eng, err := genetic.NewEngine(cfg)
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}

	res, err := eng.Run()
	if err != nil {
		t.Fatalf("engine run failed: %v", err)
	}

	var buf bytes.Buffer
	if err := res.History.ExportCSV(&buf); err != nil {
		t.Fatalf("CSV export failed: %v", err)
	}

	output := buf.String()
	if len(output) == 0 {
		t.Fatal("expected non-empty CSV output")
	}
}

func TestOnGenerationCallback(t *testing.T) {
	cfg := targetStringSetup("callback test")
	cfg.Generations = 100
	generationsObserved := 0

	cfg.OnGeneration = func(stats genetic.GenerationStats, pop genetic.Population[[]byte]) bool {
		generationsObserved++
		// Stop early at generation 5
		return stats.Generation < 5
	}

	eng, err := genetic.NewEngine(cfg)
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}

	res, err := eng.Run()
	if err != nil {
		t.Fatalf("engine run failed: %v", err)
	}

	if generationsObserved != 5 {
		t.Fatalf("expected 5 generations observed, got %d", generationsObserved)
	}

	if res.ConvergedReason != "user_callback_stop" {
		t.Fatalf("expected converged reason 'user_callback_stop', got '%s'", res.ConvergedReason)
	}
}

func TestIslandEngine(t *testing.T) {
	baseCfg := targetStringSetup("island engine model test")

	islandCfg := genetic.IslandConfig[[]byte]{
		NumIslands:           4,
		IslandPopulationSize: 30,
		EpochGenerations:     5,
		MigrationCount:       2,
		TotalGenerations:     25,
		Topology:             genetic.RingTopology,
		BaseConfig:           baseCfg,
	}

	islandEng, err := genetic.NewIslandEngine(islandCfg)
	if err != nil {
		t.Fatalf("failed to create island engine: %v", err)
	}

	res, err := islandEng.Run()
	if err != nil {
		t.Fatalf("island engine run failed: %v", err)
	}

	if len(res.FinalPopulation) != 4*30 {
		t.Fatalf("expected final population size %d, got %d", 4*30, len(res.FinalPopulation))
	}

	if res.BestIndividual.Fitness <= 0 {
		t.Fatalf("expected positive best fitness, got %f", res.BestIndividual.Fitness)
	}
}

func TestReproducibilityAcrossWorkerCounts(t *testing.T) {
	workerCounts := []int{1, 2, 4, 8}
	var baselineResult genetic.Result[[]byte]

	for i, workers := range workerCounts {
		cfg := targetStringSetup("worker count reproducibility")
		cfg.Seed = 9999
		cfg.NumWorkers = workers

		eng, err := genetic.NewEngine(cfg)
		if err != nil {
			t.Fatalf("failed to create engine with %d workers: %v", workers, err)
		}

		res, err := eng.Run()
		if err != nil {
			t.Fatalf("run failed with %d workers: %v", workers, err)
		}

		if i == 0 {
			baselineResult = res
		} else {
			if !bytes.Equal(res.BestIndividual.Genome, baselineResult.BestIndividual.Genome) {
				t.Fatalf("worker count %d mismatch in best genome: '%s' vs '%s'",
					workers, string(res.BestIndividual.Genome), string(baselineResult.BestIndividual.Genome))
			}
			if res.BestIndividual.Fitness != baselineResult.BestIndividual.Fitness {
				t.Fatalf("worker count %d mismatch in best fitness: %f vs %f",
					workers, res.BestIndividual.Fitness, baselineResult.BestIndividual.Fitness)
			}
		}
	}
}

func TestContractInvariants(t *testing.T) {
	t.Run("Worker Pool Evaluates All Individuals", func(t *testing.T) {
		cfg := targetStringSetup("eval completeness")
		cfg.NumWorkers = 4
		eng, _ := genetic.NewEngine(cfg)
		res, _ := eng.Run()

		for i, ind := range res.FinalPopulation {
			if !ind.Evaluated {
				t.Fatalf("individual at index %d was not evaluated", i)
			}
		}
	})

	t.Run("Elitism Preserves Top Individuals", func(t *testing.T) {
		pop := genetic.Population[int]{
			{Genome: 1, Fitness: 10.0, Evaluated: true},
			{Genome: 2, Fitness: 90.0, Evaluated: true},
			{Genome: 3, Fitness: 50.0, Evaluated: true},
		}

		elites := genetic.ExtractElite(pop, 2, nil)
		if len(elites) != 2 {
			t.Fatalf("expected 2 elites, got %d", len(elites))
		}
		if elites[0].Fitness != 90.0 || elites[1].Fitness != 50.0 {
			t.Fatalf("expected elites with fitness 90 and 50, got %f and %f", elites[0].Fitness, elites[1].Fitness)
		}
	})
}


