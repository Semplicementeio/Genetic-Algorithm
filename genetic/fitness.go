package genetic

import (
	"context"
	"fmt"
	"math"
)

// maxEvalGrain bounds how many individuals a worker claims at once. Small enough
// to balance uneven fitness costs, large enough to amortize the atomic counter.
const maxEvalGrain = 64

// Evaluator manages fitness calculations for a population, supporting parallel workers.
type Evaluator[T any] struct {
	fitnessFunc FitnessFunc[T]
	numWorkers  int
}

// NewEvaluator creates a new Evaluator with the given fitness function and worker count.
func NewEvaluator[T any](fn FitnessFunc[T], numWorkers int) *Evaluator[T] {
	if numWorkers < 1 {
		numWorkers = 1
	}
	return &Evaluator[T]{
		fitnessFunc: fn,
		numWorkers:  numWorkers,
	}
}

// Evaluate calculates fitness for all unevaluated individuals in the population.
// It is equivalent to EvaluateContext with context.Background().
func (e *Evaluator[T]) Evaluate(pop Population[T]) error {
	return e.EvaluateContext(context.Background(), pop)
}

// EvaluateContext calculates fitness for all unevaluated individuals, running up
// to NumWorkers fitness calls concurrently. A panic inside the fitness function is
// recovered and returned as an error wrapping ErrPanicRecovered; a NaN result is
// returned as ErrNaNFitness; cancellation returns ctx.Err(). Individuals that were
// not evaluated because of an error keep Evaluated == false.
func (e *Evaluator[T]) EvaluateContext(ctx context.Context, pop Population[T]) error {
	pending := make([]int, 0, len(pop))
	for i := range pop {
		if !pop[i].Evaluated {
			pending = append(pending, i)
		}
	}
	if len(pending) == 0 {
		return nil
	}

	grain := len(pending) / (e.numWorkers * 8)
	if grain > maxEvalGrain {
		grain = maxEvalGrain
	}

	return runParallel(ctx, len(pending), e.numWorkers, grain, func(k int) error {
		ind := &pop[pending[k]]
		fit := e.fitnessFunc(ind.Genome)
		if math.IsNaN(fit) {
			return fmt.Errorf("%w (individual %d)", ErrNaNFitness, pending[k])
		}
		ind.Fitness = fit
		ind.Evaluated = true
		return nil
	})
}
