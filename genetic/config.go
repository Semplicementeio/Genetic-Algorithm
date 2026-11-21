package genetic

import (
	"errors"
	"fmt"
	"math/rand"
)

var (
	ErrInvalidPopulationSize = errors.New("population size must be at least 2")
	ErrInvalidGenerations    = errors.New("generations must be greater than 0")
	ErrInvalidMutationRate   = errors.New("mutation rate must be between 0.0 and 1.0")
	ErrInvalidCrossoverRate  = errors.New("crossover rate must be between 0.0 and 1.0")
	ErrInvalidElitismCount   = errors.New("elitism count must be non-negative and strictly less than population size")
	ErrInvalidWorkers        = errors.New("number of workers must be at least 1")
	ErrNilFitnessFunc        = errors.New("fitness function cannot be nil")
	ErrNilGenerator          = errors.New("generator function cannot be nil")
	ErrNilSelector           = errors.New("selector cannot be nil")
	ErrNilCrossover          = errors.New("crossover operator cannot be nil")
	ErrNilMutator            = errors.New("mutator operator cannot be nil")
)

// GeneratorFunc generates a single random genome.
type GeneratorFunc[T any] func(rng *rand.Rand) T

// FitnessFunc evaluates the fitness of a genome. Higher values indicate better fitness.
type FitnessFunc[T any] func(genome T) float64

// GenomeCopierFunc creates a deep copy of a genome T. Optional for value types or slices if provided.
type GenomeCopierFunc[T any] func(genome T) T

// Config holds all parameters and operator strategies for the genetic algorithm.
type Config[T any] struct {
	// PopulationSize is the number of individuals per generation (at least 2).
	PopulationSize int
	// Generations is the number of populations evaluated, counting the initial
	// random one: Generations=N performs N-1 rounds of breeding.
	Generations   int
	MutationRate  float64
	CrossoverRate float64
	// ElitismCount individuals are copied unchanged into the next generation.
	ElitismCount int

	// NumWorkers is the parallelism of the engine (at least 1). With NumWorkers > 1,
	// fitness evaluation AND offspring creation (Selector, Crossover, Mutator and
	// GenomeCopier calls) run on several goroutines, so those functions must be safe
	// for concurrent use and must not mutate their inputs or shared state. Every
	// operator shipped in this package is. Results are identical for any NumWorkers.
	NumWorkers int

	// Seed makes a run reproducible. Seed == 0 means "pick a random seed"; the seed
	// actually used is reported in Result.Seed so any run can be replayed.
	Seed int64

	// TargetFitness stops the run as soon as the best fitness reaches it (optional).
	TargetFitness *float64
	// MaxGenerationsWithoutProgress stops the run after that many generations
	// without a new best fitness; 0 disables the check.
	MaxGenerationsWithoutProgress int

	// Observers are read-only callbacks invoked after each generation (e.g. for logging, progress bars, visualization).
	Observers []func(stats GenerationStats, pop Population[T])

	// OnGeneration is an optional control callback. If it returns false, evolution terminates early.
	OnGeneration func(stats GenerationStats, pop Population[T]) bool

	Generator    GeneratorFunc[T]
	FitnessFunc  FitnessFunc[T]
	Selector     Selector[T]
	Crossover    Crossover[T]
	Mutator      Mutator[T]
	GenomeCopier GenomeCopierFunc[T]
}

// Validate checks the configuration for invalid or inconsistent parameters.
func (c *Config[T]) Validate() error {
	if c.PopulationSize < 2 {
		return fmt.Errorf("%w: got %d", ErrInvalidPopulationSize, c.PopulationSize)
	}
	if c.Generations <= 0 {
		return fmt.Errorf("%w: got %d", ErrInvalidGenerations, c.Generations)
	}
	if c.MutationRate < 0.0 || c.MutationRate > 1.0 {
		return fmt.Errorf("%w: got %.4f", ErrInvalidMutationRate, c.MutationRate)
	}
	if c.CrossoverRate < 0.0 || c.CrossoverRate > 1.0 {
		return fmt.Errorf("%w: got %.4f", ErrInvalidCrossoverRate, c.CrossoverRate)
	}
	if c.ElitismCount < 0 || c.ElitismCount >= c.PopulationSize {
		return fmt.Errorf("%w: got %d for population size %d", ErrInvalidElitismCount, c.ElitismCount, c.PopulationSize)
	}
	if c.NumWorkers < 1 {
		return fmt.Errorf("%w: got %d", ErrInvalidWorkers, c.NumWorkers)
	}
	if c.FitnessFunc == nil {
		return ErrNilFitnessFunc
	}
	if c.Generator == nil {
		return ErrNilGenerator
	}
	if c.Selector == nil {
		return ErrNilSelector
	}
	if c.Crossover == nil {
		return ErrNilCrossover
	}
	if c.Mutator == nil {
		return ErrNilMutator
	}
	return nil
}

// DefaultConfig returns a sensible baseline configuration.
func DefaultConfig[T any]() Config[T] {
	return Config[T]{
		PopulationSize: 100,
		Generations:    100,
		MutationRate:   0.05,
		CrossoverRate:  0.80,
		ElitismCount:   2,
		NumWorkers:     1,
		Seed:           0,
	}
}
