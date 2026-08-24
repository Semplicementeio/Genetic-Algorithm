package genetic

import (
	"math/rand"
)

// Crossover defines the interface for recombination operators.
type Crossover[T any] interface {
	Cross(p1, p2 T, rng *rand.Rand) (T, T)
}

// SinglePointCrossover performs single-point crossover on slice genomes.
type SinglePointCrossover[T ~[]E, E any] struct{}

func NewSinglePointCrossover[T ~[]E, E any]() SinglePointCrossover[T, E] {
	return SinglePointCrossover[T, E]{}
}

func (sp SinglePointCrossover[T, E]) Cross(p1, p2 T, rng *rand.Rand) (T, T) {
	n := len(p1)
	if n <= 1 || len(p2) != n {
		c1 := make(T, len(p1))
		c2 := make(T, len(p2))
		copy(c1, p1)
		copy(c2, p2)
		return c1, c2
	}

	point := rng.Intn(n-1) + 1

	c1 := make(T, n)
	c2 := make(T, n)

	copy(c1[:point], p1[:point])
	copy(c1[point:], p2[point:])

	copy(c2[:point], p2[:point])
	copy(c2[point:], p1[point:])

	return c1, c2
}

// TwoPointCrossover performs two-point crossover on slice genomes.
type TwoPointCrossover[T ~[]E, E any] struct{}

func NewTwoPointCrossover[T ~[]E, E any]() TwoPointCrossover[T, E] {
	return TwoPointCrossover[T, E]{}
}

func (tp TwoPointCrossover[T, E]) Cross(p1, p2 T, rng *rand.Rand) (T, T) {
	n := len(p1)
	if n <= 2 || len(p2) != n {
		return SinglePointCrossover[T, E]{}.Cross(p1, p2, rng)
	}

	pt1 := rng.Intn(n - 1)
	pt2 := rng.Intn(n-pt1-1) + pt1 + 1

	c1 := make(T, n)
	c2 := make(T, n)

	copy(c1, p1)
	copy(c2, p2)

	for i := pt1; i < pt2; i++ {
		c1[i] = p2[i]
		c2[i] = p1[i]
	}

	return c1, c2
}

// UniformCrossover swaps elements between parents with a given swap probability.
type UniformCrossover[T ~[]E, E any] struct {
	SwapProb float64
}

func NewUniformCrossover[T ~[]E, E any](swapProb float64) UniformCrossover[T, E] {
	if swapProb <= 0.0 || swapProb >= 1.0 {
		swapProb = 0.5
	}
	return UniformCrossover[T, E]{SwapProb: swapProb}
}

func (uc UniformCrossover[T, E]) Cross(p1, p2 T, rng *rand.Rand) (T, T) {
	n := len(p1)
	c1 := make(T, n)
	c2 := make(T, n)
	copy(c1, p1)
	copy(c2, p2)

	minLen := n
	if len(p2) < minLen {
		minLen = len(p2)
	}

	for i := 0; i < minLen; i++ {
		if rng.Float64() < uc.SwapProb {
			c1[i], c2[i] = p2[i], p1[i]
		}
	}

	return c1, c2
}

// OrderCrossover (OX1) preserves relative order of elements, ideal for permutation problems like TSP.
type OrderCrossover[T ~[]E, E comparable] struct{}

func NewOrderCrossover[T ~[]E, E comparable]() OrderCrossover[T, E] {
	return OrderCrossover[T, E]{}
}

func (ox OrderCrossover[T, E]) Cross(p1, p2 T, rng *rand.Rand) (T, T) {
	n := len(p1)
	if n <= 2 || len(p2) != n {
		c1 := make(T, n)
		c2 := make(T, len(p2))
		copy(c1, p1)
		copy(c2, p2)
		return c1, c2
	}

	start := rng.Intn(n - 1)
	end := rng.Intn(n-start-1) + start + 1

	c1 := ox.oxChild(p1, p2, start, end)
	c2 := ox.oxChild(p2, p1, start, end)

	return c1, c2
}

func (ox OrderCrossover[T, E]) oxChild(p1, p2 T, start, end int) T {
	n := len(p1)
	child := make(T, n)
	inChild := make(map[E]bool, n)

	for i := start; i <= end; i++ {
		child[i] = p1[i]
		inChild[p1[i]] = true
	}

	childIdx := (end + 1) % n
	p2Idx := (end + 1) % n

	for count := 0; count < n; count++ {
		elem := p2[p2Idx]
		if !inChild[elem] {
			child[childIdx] = elem
			inChild[elem] = true
			childIdx = (childIdx + 1) % n
		}
		p2Idx = (p2Idx + 1) % n
	}

	return child
}
