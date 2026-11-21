package genetic

import (
	"math/rand"
	"sort"
)

// Selector defines the interface for selection strategies.
//
// When Config.NumWorkers > 1 the engine calls Select from several goroutines at
// once (each with its own *rand.Rand), so implementations must not mutate shared
// state. All selectors in this package are safe.
type Selector[T any] interface {
	Select(pop Population[T], rng *rand.Rand) Individual[T]
}

// PreparableSelector is an optional extension of Selector. Selection schemes that
// need per-population precomputation (cumulative weights, rankings...) implement
// it so the engine does that work once per generation instead of once per
// Select call. Prepare must return a selector bound to the given population that
// is safe for concurrent use; it must not retain or mutate the caller's state.
type PreparableSelector[T any] interface {
	Selector[T]
	Prepare(pop Population[T]) Selector[T]
}

// prepareSelector returns a per-generation selector for pop.
func prepareSelector[T any](s Selector[T], pop Population[T]) Selector[T] {
	if p, ok := s.(PreparableSelector[T]); ok {
		return p.Prepare(pop)
	}
	return s
}

// TournamentSelection selects the best individual among K randomly chosen individuals.
type TournamentSelection[T any] struct {
	K int
}

// NewTournamentSelection creates a new TournamentSelection operator with tournament size K.
func NewTournamentSelection[T any](k int) TournamentSelection[T] {
	if k < 1 {
		k = 2
	}
	return TournamentSelection[T]{K: k}
}

func (ts TournamentSelection[T]) Select(pop Population[T], rng *rand.Rand) Individual[T] {
	if len(pop) == 0 {
		return Individual[T]{}
	}

	bestIdx := rng.Intn(len(pop))
	bestFitness := pop[bestIdx].Fitness

	for i := 1; i < ts.K; i++ {
		idx := rng.Intn(len(pop))
		if pop[idx].Fitness > bestFitness {
			bestFitness = pop[idx].Fitness
			bestIdx = idx
		}
	}

	return pop[bestIdx]
}

// RouletteWheelSelection implements fitness-proportionate selection.
// Negative fitness values are shifted so that the worst individual has a
// near-zero (but positive) weight.
type RouletteWheelSelection[T any] struct{}

func NewRouletteWheelSelection[T any]() RouletteWheelSelection[T] {
	return RouletteWheelSelection[T]{}
}

// Select picks one individual. It costs O(n) per call; inside an Engine the
// selector is prepared once per generation and each pick costs O(log n).
func (rw RouletteWheelSelection[T]) Select(pop Population[T], rng *rand.Rand) Individual[T] {
	return rw.Prepare(pop).Select(pop, rng)
}

// Prepare implements PreparableSelector.
func (rw RouletteWheelSelection[T]) Prepare(pop Population[T]) Selector[T] {
	snap := &rouletteSnapshot[T]{pop: pop}
	if len(pop) == 0 {
		return snap
	}

	// Find minimum fitness to shift if fitness values are negative
	minFit := pop[0].Fitness
	for _, ind := range pop {
		if ind.Fitness < minFit {
			minFit = ind.Fitness
		}
	}
	shift := 0.0
	if minFit < 0 {
		shift = -minFit + 1e-6
	}

	snap.cumulative = make([]float64, len(pop))
	total := 0.0
	for i, ind := range pop {
		total += ind.Fitness + shift
		snap.cumulative[i] = total
	}
	snap.total = total
	return snap
}

type rouletteSnapshot[T any] struct {
	pop        Population[T]
	cumulative []float64
	total      float64
}

func (s *rouletteSnapshot[T]) Select(_ Population[T], rng *rand.Rand) Individual[T] {
	if len(s.pop) == 0 {
		return Individual[T]{}
	}
	if s.total <= 0 {
		// Uniform random fallback if all fitness values are equal or 0
		return s.pop[rng.Intn(len(s.pop))]
	}
	pick := rng.Float64() * s.total
	idx := sort.SearchFloat64s(s.cumulative, pick) // first cumulative >= pick
	if idx >= len(s.pop) {
		idx = len(s.pop) - 1
	}
	return s.pop[idx]
}

// RankSelection ranks individuals by fitness and selects with probability proportional to rank.
type RankSelection[T any] struct{}

func NewRankSelection[T any]() RankSelection[T] {
	return RankSelection[T]{}
}

// Select picks one individual. It costs O(n log n) per call; inside an Engine the
// selector is prepared once per generation and each pick costs O(log n).
func (rs RankSelection[T]) Select(pop Population[T], rng *rand.Rand) Individual[T] {
	return rs.Prepare(pop).Select(pop, rng)
}

// Prepare implements PreparableSelector.
func (rs RankSelection[T]) Prepare(pop Population[T]) Selector[T] {
	order := make([]int, len(pop))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return pop[order[a]].Fitness < pop[order[b]].Fitness })
	return &rankSnapshot[T]{pop: pop, order: order}
}

type rankSnapshot[T any] struct {
	pop   Population[T]
	order []int // population indices sorted by ascending fitness (rank 1 = worst)
}

func (s *rankSnapshot[T]) Select(_ Population[T], rng *rand.Rand) Individual[T] {
	n := len(s.pop)
	if n == 0 {
		return Individual[T]{}
	}

	// Total sum of ranks 1..N = N*(N+1)/2; pick the smallest rank r with r(r+1)/2 >= pick.
	totalRankSum := float64(n) * float64(n+1) / 2
	pick := rng.Float64() * totalRankSum
	r := sort.Search(n, func(i int) bool {
		rank := float64(i + 1)
		return rank*(rank+1)/2 >= pick
	})
	if r >= n {
		r = n - 1
	}
	return s.pop[s.order[r]]
}

// eliteSortThreshold is the largest n for which ExtractElite uses a single pass
// with a small insertion buffer (O(len(pop)*n)) instead of a full sort.
const eliteSortThreshold = 32

// ExtractElite returns clones of the top n individuals of pop, best first.
// Ties keep their original population order. The input is never reordered.
func ExtractElite[T any](pop Population[T], n int, copier GenomeCopierFunc[T]) Population[T] {
	if n <= 0 || len(pop) == 0 {
		return Population[T]{}
	}
	if n > len(pop) {
		n = len(pop)
	}

	idx := make([]int, 0, n)
	if n <= eliteSortThreshold {
		for i := range pop {
			f := pop[i].Fitness
			if len(idx) == n && !(f > pop[idx[n-1]].Fitness) {
				continue
			}
			var j int
			if len(idx) < n {
				idx = append(idx, i)
				j = len(idx) - 1
			} else {
				idx[n-1] = i
				j = n - 1
			}
			for j > 0 && pop[idx[j-1]].Fitness < f {
				idx[j], idx[j-1] = idx[j-1], idx[j]
				j--
			}
		}
	} else {
		all := make([]int, len(pop))
		for i := range all {
			all[i] = i
		}
		sort.SliceStable(all, func(a, b int) bool { return pop[all[a]].Fitness > pop[all[b]].Fitness })
		idx = all[:n]
	}

	elite := make(Population[T], n)
	for i, k := range idx {
		elite[i] = pop[k].Clone(copier)
	}
	return elite
}
