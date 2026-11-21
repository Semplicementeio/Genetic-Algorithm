package genetic_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/Semplicementeio/Genetic-Algorithm/genetic"
)

func isPermutation(g []int, n int) bool {
	if len(g) != n {
		return false
	}
	seen := make([]bool, n)
	for _, v := range g {
		if v < 0 || v >= n || seen[v] {
			return false
		}
		seen[v] = true
	}
	return true
}

func TestSlicesCrossoversPreserveLengthAndGenes(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	p1 := []int{0, 1, 2, 3, 4, 5, 6, 7}
	p2 := []int{10, 11, 12, 13, 14, 15, 16, 17}

	ops := map[string]genetic.Crossover[[]int]{
		"single":  genetic.NewSinglePointCrossover[[]int, int](),
		"two":     genetic.NewTwoPointCrossover[[]int, int](),
		"uniform": genetic.NewUniformCrossover[[]int, int](0.5),
	}
	for name, op := range ops {
		for i := 0; i < 500; i++ {
			c1, c2 := op.Cross(p1, p2, rng)
			if len(c1) != len(p1) || len(c2) != len(p2) {
				t.Fatalf("%s: length changed", name)
			}
			for j := range c1 {
				if c1[j]+c2[j] != p1[j]+p2[j] {
					t.Fatalf("%s: gene %d not conserved: %v %v", name, j, c1, c2)
				}
			}
		}
		// Parents must never be mutated or aliased.
		if p1[0] != 0 || p2[0] != 10 {
			t.Fatalf("%s mutated a parent", name)
		}
	}
}

func TestCrossoversHandleDegenerateGenomes(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	ops := map[string]genetic.Crossover[[]int]{
		"single":  genetic.NewSinglePointCrossover[[]int, int](),
		"two":     genetic.NewTwoPointCrossover[[]int, int](),
		"uniform": genetic.NewUniformCrossover[[]int, int](0.5),
		"order":   genetic.NewOrderCrossover[[]int, int](),
	}
	for name, op := range ops {
		for _, pair := range [][2][]int{{{}, {}}, {{1}, {2}}, {{1, 2}, {3, 4}}, {{1, 2, 3}, {4}}} {
			c1, c2 := op.Cross(pair[0], pair[1], rng) // must not panic
			_ = c1
			_ = c2
			_ = name
		}
	}
}

func TestOrderCrossoverAlwaysYieldsPermutations(t *testing.T) {
	rng := rand.New(rand.NewSource(8))
	ox := genetic.NewOrderCrossover[[]int, int]()
	for n := 3; n <= 12; n++ {
		p1 := rng.Perm(n)
		p2 := rng.Perm(n)
		for i := 0; i < 200; i++ {
			c1, c2 := ox.Cross(p1, p2, rng)
			if !isPermutation(c1, n) || !isPermutation(c2, n) {
				t.Fatalf("n=%d: not permutations: %v %v", n, c1, c2)
			}
		}
	}
}

func TestUniformCrossoverDefaultSwapProb(t *testing.T) {
	if genetic.NewUniformCrossover[[]int, int](-1).SwapProb != 0.5 || genetic.NewUniformCrossover[[]int, int](1).SwapProb != 0.5 {
		t.Fatal("out-of-range swap probability should default to 0.5")
	}
}

func TestMutationOperators(t *testing.T) {
	rng := rand.New(rand.NewSource(5))

	t.Run("BitFlip", func(t *testing.T) {
		g := make([]bool, 1000)
		m := genetic.NewBitFlipMutation()
		if out := m.Mutate(g, 0, rng); countTrue(out) != 0 {
			t.Fatal("rate 0 must not flip")
		}
		if out := m.Mutate(g, 1, rng); countTrue(out) != 1000 {
			t.Fatal("rate 1 must flip all")
		}
		if out := m.Mutate(g, 0.5, rng); countTrue(out) < 400 || countTrue(out) > 600 {
			t.Fatalf("rate 0.5 flipped %d/1000", countTrue(out))
		}
		if countTrue(g) != 0 {
			t.Fatal("input genome was mutated in place")
		}
	})

	t.Run("Byte", func(t *testing.T) {
		g := []byte("aaaaaaaaaa")
		out := genetic.NewByteMutation("xy").Mutate(g, 1, rng)
		for _, b := range out {
			if b != 'x' && b != 'y' {
				t.Fatalf("byte %q outside alphabet", b)
			}
		}
		if string(g) != "aaaaaaaaaa" {
			t.Fatal("input genome was mutated in place")
		}
		_ = genetic.NewByteMutation("").Mutate(g, 1, rng) // random bytes path
	})

	t.Run("Gaussian", func(t *testing.T) {
		g := make([]float64, 5000)
		out := genetic.NewGaussianMutation(0.5).Mutate(g, 1, rng)
		var sum, sq float64
		for _, v := range out {
			sum += v
			sq += v * v
		}
		mean := sum / float64(len(out))
		std := math.Sqrt(sq/float64(len(out)) - mean*mean)
		if math.Abs(mean) > 0.05 || math.Abs(std-0.5) > 0.05 {
			t.Fatalf("unexpected distribution: mean=%f std=%f", mean, std)
		}
		if genetic.NewGaussianMutation(0).StdDev != 0.1 {
			t.Fatal("non-positive stddev should default to 0.1")
		}
	})

	t.Run("Swap and Inversion keep permutations", func(t *testing.T) {
		muts := map[string]genetic.Mutator[[]int]{
			"swap":      genetic.NewSwapMutation[[]int, int](),
			"inversion": genetic.NewInversionMutation[[]int, int](),
		}
		for name, m := range muts {
			g := rng.Perm(10)
			orig := append([]int(nil), g...)
			changed := false
			for i := 0; i < 100; i++ {
				out := m.Mutate(g, 1, rng)
				if !isPermutation(out, 10) {
					t.Fatalf("%s: not a permutation: %v", name, out)
				}
				if !equalInts(out, orig) {
					changed = true
				}
			}
			if !changed {
				t.Fatalf("%s never changed the genome at rate 1", name)
			}
			if !equalInts(g, orig) {
				t.Fatalf("%s mutated the input in place", name)
			}
			if out := m.Mutate(g, 0, rng); !equalInts(out, orig) {
				t.Fatalf("%s changed genome at rate 0", name)
			}
			_ = m.Mutate([]int{1}, 1, rng) // must not hang or panic
			_ = m.Mutate(nil, 1, rng)
		}
	})
}

func TestSelectionOperators(t *testing.T) {
	pop := genetic.Population[int]{
		{Genome: 0, Fitness: -5}, {Genome: 1, Fitness: 1}, {Genome: 2, Fitness: 2}, {Genome: 3, Fitness: 50},
	}
	selectors := map[string]genetic.Selector[int]{
		"tournament": genetic.NewTournamentSelection[int](3),
		"roulette":   genetic.NewRouletteWheelSelection[int](),
		"rank":       genetic.NewRankSelection[int](),
	}
	for name, sel := range selectors {
		rng := rand.New(rand.NewSource(9))
		counts := make([]int, len(pop))
		for i := 0; i < 4000; i++ {
			counts[sel.Select(pop, rng).Genome]++
		}
		if counts[3] <= counts[0] {
			t.Fatalf("%s: fittest (%d picks) should beat worst (%d picks): %v", name, counts[3], counts[0], counts)
		}
		for i, c := range counts {
			// Rank selection gives every individual non-zero probability.
			// (Roulette shifts negatives so the worst gets ~0 weight by design.)
			if name == "rank" && c == 0 {
				t.Fatalf("%s: individual %d never selected: %v", name, i, counts)
			}
		}
		if got := sel.Select(nil, rng); got.Evaluated || got.Fitness != 0 {
			t.Fatalf("%s: empty population should yield zero Individual", name)
		}
	}

	t.Run("roulette uniform fallback", func(t *testing.T) {
		flat := genetic.Population[int]{{Genome: 0}, {Genome: 1}, {Genome: 2}}
		rng := rand.New(rand.NewSource(1))
		seen := map[int]bool{}
		for i := 0; i < 200; i++ {
			seen[genetic.NewRouletteWheelSelection[int]().Select(flat, rng).Genome] = true
		}
		if len(seen) != 3 {
			t.Fatalf("zero-fitness population should select uniformly, saw %v", seen)
		}
	})

	t.Run("tournament K defaults", func(t *testing.T) {
		if genetic.NewTournamentSelection[int](0).K != 2 {
			t.Fatal("K<1 should default to 2")
		}
	})

	t.Run("elite edge cases", func(t *testing.T) {
		if len(genetic.ExtractElite(pop, 0, nil)) != 0 || len(genetic.ExtractElite(genetic.Population[int]{}, 3, nil)) != 0 {
			t.Fatal("expected empty elite")
		}
		if len(genetic.ExtractElite(pop, 99, nil)) != len(pop) {
			t.Fatal("n larger than population should be capped")
		}
		if pop[0].Fitness != -5 {
			t.Fatal("ExtractElite must not reorder the input population")
		}
	})
}

func TestDefaultConfigAndEvaluatorModes(t *testing.T) {
	cfg := genetic.DefaultConfig[[]byte]()
	if cfg.PopulationSize != 100 || cfg.NumWorkers != 1 || cfg.ElitismCount != 2 {
		t.Fatalf("unexpected defaults: %+v", cfg)
	}

	for _, workers := range []int{0, 1, 6} {
		pop := make(genetic.Population[int], 25)
		for i := range pop {
			pop[i].Genome = i
		}
		pop[3] = genetic.Individual[int]{Genome: 3, Fitness: 999, Evaluated: true}
		genetic.NewEvaluator(func(g int) float64 { return float64(g) * 2 }, workers).Evaluate(pop)
		for i, ind := range pop {
			want := float64(i) * 2
			if i == 3 {
				want = 999 // already evaluated individuals are not re-evaluated
			}
			if !ind.Evaluated || ind.Fitness != want {
				t.Fatalf("workers=%d idx=%d: %+v", workers, i, ind)
			}
		}
	}
}

func TestConfigValidationBranches(t *testing.T) {
	base := func() genetic.Config[[]byte] { return targetStringSetup("validate") }
	cases := map[string]func(*genetic.Config[[]byte]){
		"generations":  func(c *genetic.Config[[]byte]) { c.Generations = 0 },
		"crossover":    func(c *genetic.Config[[]byte]) { c.CrossoverRate = -0.1 },
		"elitism":      func(c *genetic.Config[[]byte]) { c.ElitismCount = c.PopulationSize },
		"workers":      func(c *genetic.Config[[]byte]) { c.NumWorkers = 0 },
		"nil gen":      func(c *genetic.Config[[]byte]) { c.Generator = nil },
		"nil selector": func(c *genetic.Config[[]byte]) { c.Selector = nil },
		"nil cross":    func(c *genetic.Config[[]byte]) { c.Crossover = nil },
		"nil mutator":  func(c *genetic.Config[[]byte]) { c.Mutator = nil },
	}
	for name, mut := range cases {
		cfg := base()
		mut(&cfg)
		if err := cfg.Validate(); err == nil {
			t.Fatalf("%s: expected validation error", name)
		}
	}
}

func TestPopulationHelpersAndStats(t *testing.T) {
	var empty genetic.Population[int]
	if empty.AverageFitness() != 0 || genetic.CalculateStdDev(empty, 0) != 0 {
		t.Fatal("empty population stats should be zero")
	}
	pop := genetic.Population[int]{{Fitness: 1}, {Fitness: 3}}
	if pop.Best().Fitness != 3 || pop.Worst().Fitness != 1 || pop.AverageFitness() != 2 {
		t.Fatal("unexpected best/worst/avg")
	}
	if got := genetic.CalculateStdDev(pop, 2); got != 1 {
		t.Fatalf("stddev = %f, want 1", got)
	}
}

func TestEngineWithoutMutationAndCrossover(t *testing.T) {
	cfg := targetStringSetup("static")
	cfg.MutationRate = 0
	cfg.CrossoverRate = 0
	eng, _ := genetic.NewEngine(cfg)
	if _, err := eng.Run(); err != nil {
		t.Fatal(err)
	}
}

func countTrue(b []bool) int {
	n := 0
	for _, v := range b {
		if v {
			n++
		}
	}
	return n
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
