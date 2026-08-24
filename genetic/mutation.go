package genetic

import (
	"math/rand"
)

// Mutator defines the interface for mutation operators.
type Mutator[T any] interface {
	Mutate(g T, rate float64, rng *rand.Rand) T
}

// BitFlipMutation flips boolean values with probability equal to mutation rate.
type BitFlipMutation struct{}

func NewBitFlipMutation() BitFlipMutation {
	return BitFlipMutation{}
}

func (bf BitFlipMutation) Mutate(g []bool, rate float64, rng *rand.Rand) []bool {
	mutated := make([]bool, len(g))
	copy(mutated, g)

	for i := range mutated {
		if rng.Float64() < rate {
			mutated[i] = !mutated[i]
		}
	}
	return mutated
}

// ByteMutation mutates byte slices using a provided alphabet or random bytes.
type ByteMutation struct {
	Alphabet string
}

func NewByteMutation(alphabet string) ByteMutation {
	return ByteMutation{Alphabet: alphabet}
}

func (bm ByteMutation) Mutate(g []byte, rate float64, rng *rand.Rand) []byte {
	mutated := make([]byte, len(g))
	copy(mutated, g)

	for i := range mutated {
		if rng.Float64() < rate {
			if len(bm.Alphabet) > 0 {
				mutated[i] = bm.Alphabet[rng.Intn(len(bm.Alphabet))]
			} else {
				mutated[i] = byte(rng.Intn(256))
			}
		}
	}
	return mutated
}

// GaussianMutation adds normal noise (mean 0, StdDev) to float64 slices.
type GaussianMutation struct {
	StdDev float64
}

func NewGaussianMutation(stdDev float64) GaussianMutation {
	if stdDev <= 0 {
		stdDev = 0.1
	}
	return GaussianMutation{StdDev: stdDev}
}

func (gm GaussianMutation) Mutate(g []float64, rate float64, rng *rand.Rand) []float64 {
	mutated := make([]float64, len(g))
	copy(mutated, g)

	for i := range mutated {
		if rng.Float64() < rate {
			mutated[i] += rng.NormFloat64() * gm.StdDev
		}
	}
	return mutated
}

// SwapMutation swaps two random elements in a slice genome.
type SwapMutation[T ~[]E, E any] struct{}

func NewSwapMutation[T ~[]E, E any]() SwapMutation[T, E] {
	return SwapMutation[T, E]{}
}

func (sm SwapMutation[T, E]) Mutate(g T, rate float64, rng *rand.Rand) T {
	mutated := make(T, len(g))
	copy(mutated, g)

	if len(mutated) < 2 {
		return mutated
	}

	if rng.Float64() < rate {
		idx1 := rng.Intn(len(mutated))
		idx2 := rng.Intn(len(mutated))
		for idx1 == idx2 {
			idx2 = rng.Intn(len(mutated))
		}
		mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
	}

	return mutated
}

// InversionMutation inverts a sub-slice in a slice genome.
type InversionMutation[T ~[]E, E any] struct{}

func NewInversionMutation[T ~[]E, E any]() InversionMutation[T, E] {
	return InversionMutation[T, E]{}
}

func (im InversionMutation[T, E]) Mutate(g T, rate float64, rng *rand.Rand) T {
	mutated := make(T, len(g))
	copy(mutated, g)

	if len(mutated) < 2 {
		return mutated
	}

	if rng.Float64() < rate {
		i := rng.Intn(len(mutated) - 1)
		j := rng.Intn(len(mutated)-i-1) + i + 1

		for i < j {
			mutated[i], mutated[j] = mutated[j], mutated[i]
			i++
			j--
		}
	}

	return mutated
}
