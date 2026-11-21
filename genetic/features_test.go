package genetic_test

import (
	"context"
	"errors"
	"math"
	"math/rand"
	"sort"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Semplicementeio/Genetic-Algorithm/genetic"
)

func TestRunContextCancellationReturnsPartialResult(t *testing.T) {
	for _, workers := range []int{1, 4} {
		cfg := targetStringSetup("context cancellation")
		cfg.NumWorkers = workers
		cfg.Generations = 1_000_000

		ctx, cancel := context.WithCancel(context.Background())
		cfg.Observers = append(cfg.Observers, func(s genetic.GenerationStats, _ genetic.Population[[]byte]) {
			if s.Generation == 10 {
				cancel()
			}
		})

		eng, _ := genetic.NewEngine(cfg)
		res, err := eng.RunContext(ctx)
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("workers=%d: expected context.Canceled, got %v", workers, err)
		}
		if res.ConvergedReason != genetic.StopContextCanceled || res.GenerationsEvaluated < 10 || res.GenerationsEvaluated > 11 {
			t.Fatalf("workers=%d: unexpected partial result: reason=%s gens=%d", workers, res.ConvergedReason, res.GenerationsEvaluated)
		}
		if len(res.FinalPopulation) != cfg.PopulationSize || res.BestIndividual.Genome == nil {
			t.Fatal("partial result must hold a complete, evaluated population")
		}
		for _, ind := range res.FinalPopulation {
			if !ind.Evaluated {
				t.Fatal("final population contains unevaluated individuals")
			}
		}
	}
}

func TestRunContextDeadlineInterruptsSlowFitness(t *testing.T) {
	cfg := targetStringSetup("deadline")
	cfg.NumWorkers = 4
	cfg.Generations = 1_000_000
	cfg.FitnessFunc = func([]byte) float64 { time.Sleep(time.Millisecond); return 1 }

	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
	defer cancel()

	start := time.Now()
	eng, _ := genetic.NewEngine(cfg)
	res, err := eng.RunContext(ctx)
	if !errors.Is(err, context.DeadlineExceeded) || res.ConvergedReason != genetic.StopContextCanceled {
		t.Fatalf("err=%v reason=%s", err, res.ConvergedReason)
	}
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Fatalf("cancellation took %v", elapsed)
	}
}

func TestRunContextAlreadyCanceled(t *testing.T) {
	cfg := targetStringSetup("pre-canceled")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	eng, _ := genetic.NewEngine(cfg)
	if _, err := eng.RunContext(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
}

func TestIslandRunContextCancellation(t *testing.T) {
	cfg := islandSetup("island cancel", 5, 100000, genetic.RingTopology)
	ctx, cancel := context.WithCancel(context.Background())
	cfg.BaseConfig.OnGeneration = func(s genetic.GenerationStats, _ genetic.Population[[]byte]) bool {
		if s.Generation >= 10 {
			cancel()
		}
		return true
	}
	eng, _ := genetic.NewIslandEngine(cfg)
	res, err := eng.RunContext(ctx)
	if !errors.Is(err, context.Canceled) || res.ConvergedReason != genetic.StopContextCanceled {
		t.Fatalf("err=%v reason=%s", err, res.ConvergedReason)
	}
	if len(res.FinalPopulation) != 4*20 || res.BestIndividual.Genome == nil {
		t.Fatal("expected a usable partial result")
	}
}

func TestFitnessPanicIsReturnedAsError(t *testing.T) {
	for _, workers := range []int{1, 4} {
		cfg := targetStringSetup("panic recovery")
		cfg.NumWorkers = workers
		var calls atomic.Int64
		cfg.FitnessFunc = func(g []byte) float64 {
			if calls.Add(1) == 150 {
				panic("boom")
			}
			return 1
		}
		eng, _ := genetic.NewEngine(cfg)
		_, err := eng.Run()
		if !errors.Is(err, genetic.ErrPanicRecovered) {
			t.Fatalf("workers=%d: expected ErrPanicRecovered, got %v", workers, err)
		}
	}
}

func TestOperatorPanicIsReturnedAsError(t *testing.T) {
	cfg := targetStringSetup("operator panic")
	cfg.NumWorkers = 3
	cfg.Mutator = panicMutator{}
	eng, _ := genetic.NewEngine(cfg)
	if _, err := eng.Run(); !errors.Is(err, genetic.ErrPanicRecovered) {
		t.Fatalf("expected ErrPanicRecovered, got %v", err)
	}
}

type panicMutator struct{}

func (panicMutator) Mutate([]byte, float64, *rand.Rand) []byte { panic("mutator exploded") }

func TestNaNFitnessIsRejected(t *testing.T) {
	cfg := targetStringSetup("nan")
	cfg.FitnessFunc = func([]byte) float64 { return math.NaN() }
	eng, _ := genetic.NewEngine(cfg)
	if _, err := eng.Run(); !errors.Is(err, genetic.ErrNaNFitness) {
		t.Fatalf("expected ErrNaNFitness, got %v", err)
	}
}

func TestRunIsRepeatableAndReportsSeed(t *testing.T) {
	cfg := targetStringSetup("repeatable")
	cfg.Seed = 0 // random
	eng, _ := genetic.NewEngine(cfg)
	a, _ := eng.Run()
	b, _ := eng.Run()
	if a.Seed == 0 || a.Seed != b.Seed || string(a.BestIndividual.Genome) != string(b.BestIndividual.Genome) {
		t.Fatalf("repeated Run on one engine must replay the same run (seed %d vs %d)", a.Seed, b.Seed)
	}

	// A random run can be replayed from Result.Seed with a fresh engine.
	cfg.Seed = a.Seed
	replay, _ := genetic.NewEngine(cfg)
	c, _ := replay.Run()
	if string(c.BestIndividual.Genome) != string(a.BestIndividual.Genome) || len(c.History) != len(a.History) {
		t.Fatal("replaying Result.Seed did not reproduce the run")
	}
}

func TestEngineConcurrentRunsAreSafe(t *testing.T) {
	cfg := targetStringSetup("concurrent runs")
	cfg.Seed = 5
	eng, _ := genetic.NewEngine(cfg)
	results := make(chan genetic.Result[[]byte], 4)
	for i := 0; i < 4; i++ {
		go func() { r, _ := eng.Run(); results <- r }()
	}
	first := <-results
	for i := 0; i < 3; i++ {
		if r := <-results; string(r.BestIndividual.Genome) != string(first.BestIndividual.Genome) {
			t.Fatal("concurrent runs diverged")
		}
	}
}

func TestStopReasonDetails(t *testing.T) {
	cfg := targetStringSetup("stop reasons")
	target := 5.0
	cfg.TargetFitness = &target
	eng, _ := genetic.NewEngine(cfg)
	res, _ := eng.Run()
	if res.ConvergedReason != genetic.StopTargetFitness || res.StopDetail == "" {
		t.Fatalf("got %q / %q", res.ConvergedReason, res.StopDetail)
	}

	cfg = targetStringSetup("stop reasons")
	cfg.FitnessFunc = func([]byte) float64 { return 1 }
	cfg.MaxGenerationsWithoutProgress = 2
	eng, _ = genetic.NewEngine(cfg)
	res, _ = eng.Run()
	if res.ConvergedReason != genetic.StopStalled || res.StopDetail == "" {
		t.Fatalf("got %q / %q", res.ConvergedReason, res.StopDetail)
	}

	cfg = targetStringSetup("stop reasons")
	cfg.Generations = 3
	eng, _ = genetic.NewEngine(cfg)
	if res, _ = eng.Run(); res.ConvergedReason != genetic.StopMaxGenerations {
		t.Fatalf("got %q", res.ConvergedReason)
	}
}

func TestPreparedSelectorsMatchUnprepared(t *testing.T) {
	pop := make(genetic.Population[int], 57)
	r := rand.New(rand.NewSource(1))
	for i := range pop {
		pop[i] = genetic.Individual[int]{Genome: i, Fitness: r.NormFloat64() * 10, Evaluated: true}
	}
	pop[10].Fitness = pop[11].Fitness // a tie

	selectors := map[string]genetic.PreparableSelector[int]{
		"roulette": genetic.NewRouletteWheelSelection[int](),
		"rank":     genetic.NewRankSelection[int](),
	}
	for name, sel := range selectors {
		prepared := sel.Prepare(pop)
		rngA := rand.New(rand.NewSource(77))
		rngB := rand.New(rand.NewSource(77))
		for i := 0; i < 2000; i++ {
			if a, b := sel.Select(pop, rngA), prepared.Select(pop, rngB); a.Genome != b.Genome {
				t.Fatalf("%s: prepared and unprepared selectors diverged at draw %d", name, i)
			}
		}
	}
}

func TestPreparedRankSelectionDistribution(t *testing.T) {
	n := 10
	pop := make(genetic.Population[int], n)
	for i := range pop {
		pop[i] = genetic.Individual[int]{Genome: i, Fitness: float64(i)}
	}
	sel := genetic.NewRankSelection[int]().Prepare(pop)
	rng := rand.New(rand.NewSource(3))
	counts := make([]float64, n)
	const draws = 220000
	for i := 0; i < draws; i++ {
		counts[sel.Select(pop, rng).Genome]++
	}
	total := float64(n * (n + 1) / 2)
	for i, c := range counts {
		want := float64(i+1) / total
		if got := c / draws; math.Abs(got-want) > 0.01 {
			t.Fatalf("rank %d selected with p=%.4f, want %.4f", i+1, got, want)
		}
	}
}

func TestExtractEliteMatchesFullSort(t *testing.T) {
	rng := rand.New(rand.NewSource(12))
	for trial := 0; trial < 200; trial++ {
		size := 1 + rng.Intn(120)
		pop := make(genetic.Population[int], size)
		for i := range pop {
			pop[i] = genetic.Individual[int]{Genome: i, Fitness: float64(rng.Intn(15)), Evaluated: true} // many ties
		}
		n := 1 + rng.Intn(size+5)

		want := append(genetic.Population[int](nil), pop...)
		sort.SliceStable(want, func(a, b int) bool { return want[a].Fitness > want[b].Fitness })
		if n > size {
			want = want[:size]
		} else {
			want = want[:n]
		}

		got := genetic.ExtractElite(pop, n, nil)
		if len(got) != len(want) {
			t.Fatalf("size=%d n=%d: got %d elites, want %d", size, n, len(got), len(want))
		}
		for i := range got {
			if got[i].Genome != want[i].Genome {
				t.Fatalf("size=%d n=%d: elite %d is genome %d, want %d (stable order)", size, n, i, got[i].Genome, want[i].Genome)
			}
		}
	}
}

func TestParallelBreedingIsDeterministicAcrossWorkers(t *testing.T) {
	// Operators run concurrently with NumWorkers > 1; use all operator kinds under -race.
	run := func(workers int) genetic.Result[[]float64] {
		cfg := genetic.Config[[]float64]{
			PopulationSize: 300, Generations: 15, MutationRate: 0.1, CrossoverRate: 0.9,
			ElitismCount: 3, NumWorkers: workers, Seed: 31,
			Generator: func(r *rand.Rand) []float64 {
				g := make([]float64, 8)
				for i := range g {
					g[i] = r.Float64()*10 - 5
				}
				return g
			},
			FitnessFunc: func(g []float64) float64 {
				s := 0.0
				for _, x := range g {
					s -= x * x
				}
				return s
			},
			Selector:  genetic.NewRankSelection[[]float64](),
			Crossover: genetic.NewTwoPointCrossover[[]float64, float64](),
			Mutator:   genetic.NewGaussianMutation(0.3),
		}
		eng, err := genetic.NewEngine(cfg)
		if err != nil {
			t.Fatal(err)
		}
		res, err := eng.Run()
		if err != nil {
			t.Fatal(err)
		}
		return res
	}
	base := run(1)
	for _, w := range []int{2, 3, 7, 16} {
		got := run(w)
		for i := range base.History {
			if got.History[i].BestFitness != base.History[i].BestFitness || got.History[i].AvgFitness != base.History[i].AvgFitness {
				t.Fatalf("workers=%d diverges from workers=1 at generation %d", w, i+1)
			}
		}
	}
}
