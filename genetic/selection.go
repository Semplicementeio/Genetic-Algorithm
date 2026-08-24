package genetic

import (
	"math/rand"
	"sort"
)

// Selector defines the interface for selection strategies.
type Selector[T any] interface {
	Select(pop Population[T], rng *rand.Rand) Individual[T]
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
type RouletteWheelSelection[T any] struct{}

func NewRouletteWheelSelection[T any]() RouletteWheelSelection[T] {
	return RouletteWheelSelection[T]{}
}

func (rw RouletteWheelSelection[T]) Select(pop Population[T], rng *rand.Rand) Individual[T] {
	if len(pop) == 0 {
		return Individual[T]{}
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

	totalFitness := 0.0
	for _, ind := range pop {
		totalFitness += ind.Fitness + shift
	}

	if totalFitness <= 0 {
		// Uniform random fallback if all fitness values are equal or 0
		return pop[rng.Intn(len(pop))]
	}

	pick := rng.Float64() * totalFitness
	current := 0.0

	for _, ind := range pop {
		current += ind.Fitness + shift
		if current >= pick {
			return ind
		}
	}

	return pop[len(pop)-1]
}

// RankSelection ranks individuals by fitness and selects with probability proportional to rank.
type RankSelection[T any] struct{}

func NewRankSelection[T any]() RankSelection[T] {
	return RankSelection[T]{}
}

func (rs RankSelection[T]) Select(pop Population[T], rng *rand.Rand) Individual[T] {
	if len(pop) == 0 {
		return Individual[T]{}
	}

	n := len(pop)
	// Create indexed array to sort by fitness
	type ranked struct {
		index int
		fit   float64
	}
	items := make([]ranked, n)
	for i, ind := range pop {
		items[i] = ranked{index: i, fit: ind.Fitness}
	}

	sort.Slice(items, func(i, j int) bool {
		return items[i].fit < items[j].fit
	})

	// Total sum of ranks 1..N = N*(N+1)/2
	totalRankSum := float64(n * (n + 1) / 2)
	pick := rng.Float64() * totalRankSum

	currentSum := 0.0
	for rank := 1; rank <= n; rank++ {
		currentSum += float64(rank)
		if currentSum >= pick {
			return pop[items[rank-1].index]
		}
	}

	return pop[items[n-1].index]
}

// ExtractElite extracts the top N individuals from a population.
func ExtractElite[T any](pop Population[T], n int, copier GenomeCopierFunc[T]) Population[T] {
	if n <= 0 || len(pop) == 0 {
		return Population[T]{}
	}
	if n > len(pop) {
		n = len(pop)
	}

	sorted := make(Population[T], len(pop))
	copy(sorted, pop)

	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Fitness > sorted[j].Fitness
	})

	elite := make(Population[T], n)
	for i := 0; i < n; i++ {
		elite[i] = sorted[i].Clone(copier)
	}
	return elite
}
