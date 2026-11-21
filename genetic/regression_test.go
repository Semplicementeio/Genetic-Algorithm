package genetic_test

import (
	"errors"
	"math/rand"
	"testing"
	"time"

	"github.com/Semplicementeio/Genetic-Algorithm/genetic"
)

func islandSetup(target string, epoch, total int, topo genetic.Topology) genetic.IslandConfig[[]byte] {
	return genetic.IslandConfig[[]byte]{
		NumIslands:           4,
		IslandPopulationSize: 20,
		EpochGenerations:     epoch,
		MigrationCount:       1,
		TotalGenerations:     total,
		Topology:             topo,
		BaseConfig:           targetStringSetup(target),
	}
}

func TestHistoryDurationCoversBreedingAndEvaluation(t *testing.T) {
	cfg := targetStringSetup("duration check")
	cfg.NumWorkers = 1
	cfg.Generations = 4
	cfg.FitnessFunc = func(g []byte) float64 {
		time.Sleep(100 * time.Microsecond)
		return 1
	}

	eng, _ := genetic.NewEngine(cfg)
	res, err := eng.Run()
	if err != nil {
		t.Fatal(err)
	}

	// Every generation evaluates ~97 unevaluated individuals at >=100µs each.
	for _, h := range res.History {
		if h.Duration < 5*time.Millisecond {
			t.Fatalf("generation %d duration %v is implausibly small; breeding/evaluation not measured", h.Generation, h.Duration)
		}
	}
}

func TestStallDetectionDoesNotCountInitialGeneration(t *testing.T) {
	cfg := targetStringSetup("stall check")
	cfg.Generations = 50
	cfg.MaxGenerationsWithoutProgress = 3
	// Constant fitness: no progress is ever possible.
	cfg.FitnessFunc = func([]byte) float64 { return 1 }

	eng, _ := genetic.NewEngine(cfg)
	res, err := eng.Run()
	if err != nil {
		t.Fatal(err)
	}
	// Generation 1 is the baseline; stall counter hits 3 at generation 4.
	if res.GenerationsEvaluated != 4 {
		t.Fatalf("expected 4 generations, got %d (%s)", res.GenerationsEvaluated, res.ConvergedReason)
	}
}

func TestGenerationsMeansEvaluatedPopulations(t *testing.T) {
	cfg := targetStringSetup("generation count")
	cfg.Generations = 7
	eng, _ := genetic.NewEngine(cfg)
	res, _ := eng.Run()
	if res.GenerationsEvaluated != 7 || len(res.History) != 7 {
		t.Fatalf("expected 7 generations, got %d / history %d", res.GenerationsEvaluated, len(res.History))
	}
}

func TestIslandSingleGenerationEpochsStillEvolve(t *testing.T) {
	// Regression: EpochGenerations=1 used to perform zero breeding.
	cfg := islandSetup("abcdefghij", 1, 60, genetic.RingTopology)
	cfg.BaseConfig.Seed = 5
	eng, err := genetic.NewIslandEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	res, err := eng.Run()
	if err != nil {
		t.Fatal(err)
	}
	first, last := res.History[0], res.History[len(res.History)-1]
	if last.AvgFitness <= first.AvgFitness {
		t.Fatalf("population did not improve: avg %f -> %f", first.AvgFitness, last.AvgFitness)
	}
}

func TestIslandGenerationAccounting(t *testing.T) {
	cfg := islandSetup("accounting", 10, 25, genetic.RingTopology)
	eng, _ := genetic.NewIslandEngine(cfg)
	res, err := eng.Run()
	if err != nil {
		t.Fatal(err)
	}
	// 25 total with epochs of 10 -> 10, 10, 5 (no silent truncation).
	if res.GenerationsEvaluated != 25 || len(res.History) != 3 {
		t.Fatalf("expected 25 generations in 3 epochs, got %d in %d", res.GenerationsEvaluated, len(res.History))
	}
	last := res.History[2]
	if last.Generation != 25 || last.WorstFitness > last.AvgFitness || last.StdDev < 0 {
		t.Fatalf("unexpected final stats: %+v", last)
	}
	if len(res.FinalPopulation) != 4*20 {
		t.Fatalf("unexpected final population size %d", len(res.FinalPopulation))
	}
}

func TestIslandCallbacksAreSerializedAndGlobal(t *testing.T) {
	// Run with -race: callbacks must not be invoked concurrently.
	cfg := islandSetup("callbacks", 5, 50, genetic.RingTopology)
	calls := 0
	cfg.BaseConfig.Observers = []func(genetic.GenerationStats, genetic.Population[[]byte]){
		func(genetic.GenerationStats, genetic.Population[[]byte]) { calls++ },
	}
	cfg.BaseConfig.OnGeneration = func(s genetic.GenerationStats, pop genetic.Population[[]byte]) bool {
		if len(pop) != 4*20 {
			t.Errorf("expected merged population of 80, got %d", len(pop))
		}
		return s.Generation < 15
	}

	eng, _ := genetic.NewIslandEngine(cfg)
	res, err := eng.Run()
	if err != nil {
		t.Fatal(err)
	}
	if calls != 3 || res.ConvergedReason != "user_callback_stop" || res.GenerationsEvaluated != 15 {
		t.Fatalf("calls=%d reason=%s gens=%d", calls, res.ConvergedReason, res.GenerationsEvaluated)
	}
}

func TestIslandTargetFitnessStopsGlobally(t *testing.T) {
	cfg := islandSetup("target", 5, 500, genetic.RingTopology)
	target := 3.0
	cfg.BaseConfig.TargetFitness = &target
	cfg.BaseConfig.Seed = 3
	eng, _ := genetic.NewIslandEngine(cfg)
	res, _ := eng.Run()
	if res.BestIndividual.Fitness < target || res.GenerationsEvaluated >= 500 {
		t.Fatalf("expected early stop at target, got fitness %f after %d gens (%s)",
			res.BestIndividual.Fitness, res.GenerationsEvaluated, res.ConvergedReason)
	}
}

func TestIslandReproducibleForAllTopologies(t *testing.T) {
	for _, topo := range []genetic.Topology{genetic.RingTopology, genetic.RandomTopology, genetic.FullyConnectedTopology} {
		run := func() genetic.Result[[]byte] {
			cfg := islandSetup("island reproducibility", 4, 20, topo)
			cfg.BaseConfig.Seed = 77
			eng, err := genetic.NewIslandEngine(cfg)
			if err != nil {
				t.Fatal(err)
			}
			res, err := eng.Run()
			if err != nil {
				t.Fatal(err)
			}
			return res
		}
		a, b := run(), run()
		if string(a.BestIndividual.Genome) != string(b.BestIndividual.Genome) || len(a.History) != len(b.History) {
			t.Fatalf("topology %d not reproducible", topo)
		}
		for i := range a.History {
			if a.History[i].AvgFitness != b.History[i].AvgFitness {
				t.Fatalf("topology %d: history diverges at epoch %d", topo, i)
			}
		}
	}
}

func TestIslandFullyConnectedMigration(t *testing.T) {
	// Island 0 holds a unique champion; after one migration every island must receive it.
	cfg := islandSetup("champion!", 1, 2, genetic.FullyConnectedTopology)
	cfg.BaseConfig.ElitismCount = 1
	cfg.BaseConfig.MutationRate = 0
	cfg.BaseConfig.CrossoverRate = 0
	cfg.BaseConfig.Seed = 11
	eng, _ := genetic.NewIslandEngine(cfg)
	res, err := eng.Run()
	if err != nil {
		t.Fatal(err)
	}
	// After epoch 1 a migration happens; epoch 2 then runs with the migrants present.
	// With no mutation/crossover, the global best genome must appear on every island.
	best := string(res.BestIndividual.Genome)
	islands := 0
	for i := 0; i < 4; i++ {
		for _, ind := range res.FinalPopulation[i*20 : (i+1)*20] {
			if string(ind.Genome) == best {
				islands++
				break
			}
		}
	}
	if islands != 4 {
		t.Fatalf("best genome %q present on %d/4 islands, expected all", best, islands)
	}
}

func TestIslandConfigValidation(t *testing.T) {
	t.Run("unknown topology", func(t *testing.T) {
		cfg := islandSetup("x", 5, 10, genetic.Topology(99))
		if _, err := genetic.NewIslandEngine(cfg); !errors.Is(err, genetic.ErrInvalidTopology) {
			t.Fatalf("expected ErrInvalidTopology, got %v", err)
		}
	})
	t.Run("fully connected overflows island", func(t *testing.T) {
		cfg := islandSetup("x", 5, 10, genetic.FullyConnectedTopology)
		cfg.MigrationCount = 7 // 7 * 3 = 21 >= 20
		if _, err := genetic.NewIslandEngine(cfg); !errors.Is(err, genetic.ErrInvalidMigrationCount) {
			t.Fatalf("expected ErrInvalidMigrationCount, got %v", err)
		}
	})
	t.Run("bad counts", func(t *testing.T) {
		for _, mut := range []func(*genetic.IslandConfig[[]byte]){
			func(c *genetic.IslandConfig[[]byte]) { c.NumIslands = 1 },
			func(c *genetic.IslandConfig[[]byte]) { c.EpochGenerations = 0 },
			func(c *genetic.IslandConfig[[]byte]) { c.MigrationCount = 20 },
			func(c *genetic.IslandConfig[[]byte]) { c.TotalGenerations = 0 },
			func(c *genetic.IslandConfig[[]byte]) { c.IslandPopulationSize = 1 },
		} {
			cfg := islandSetup("x", 5, 10, genetic.RingTopology)
			mut(&cfg)
			if _, err := genetic.NewIslandEngine(cfg); err == nil {
				t.Fatal("expected validation error")
			}
		}
	})
}

func TestTwoPointCrossoverCanSwapEveryGene(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	tp := genetic.NewTwoPointCrossover[[]int, int]()
	p1 := []int{0, 0, 0, 0, 0, 0}
	p2 := []int{1, 1, 1, 1, 1, 1}
	swapped := make([]int, len(p1))
	for i := 0; i < 5000; i++ {
		c1, c2 := tp.Cross(p1, p2, rng)
		for j := range c1 {
			if c1[j]+c2[j] != 1 {
				t.Fatalf("children must be complementary: %v %v", c1, c2)
			}
			swapped[j] += c1[j]
		}
	}
	for j, n := range swapped {
		if n == 0 {
			t.Fatalf("gene %d was never exchanged: %v", j, swapped)
		}
	}
}
