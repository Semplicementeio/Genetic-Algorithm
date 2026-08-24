package genetic

import (
	"sync"
)

// Evaluator manages fitness calculations for a population, supporting parallel worker pools.
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
func (e *Evaluator[T]) Evaluate(pop Population[T]) {
	if e.numWorkers <= 1 {
		e.evaluateSequential(pop)
	} else {
		e.evaluateParallel(pop)
	}
}

func (e *Evaluator[T]) evaluateSequential(pop Population[T]) {
	for i := range pop {
		if !pop[i].Evaluated {
			pop[i].Fitness = e.fitnessFunc(pop[i].Genome)
			pop[i].Evaluated = true
		}
	}
}

func (e *Evaluator[T]) evaluateParallel(pop Population[T]) {
	type job struct {
		index  int
		genome T
	}
	type result struct {
		index   int
		fitness float64
	}

	unevaluatedCount := 0
	for i := range pop {
		if !pop[i].Evaluated {
			unevaluatedCount++
		}
	}

	if unevaluatedCount == 0 {
		return
	}

	jobs := make(chan job, unevaluatedCount)
	results := make(chan result, unevaluatedCount)

	workers := e.numWorkers
	if workers > unevaluatedCount {
		workers = unevaluatedCount
	}

	var wg sync.WaitGroup
	wg.Add(workers)

	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			for j := range jobs {
				fit := e.fitnessFunc(j.genome)
				results <- result{index: j.index, fitness: fit}
			}
		}()
	}

	for i := range pop {
		if !pop[i].Evaluated {
			jobs <- job{index: i, genome: pop[i].Genome}
		}
	}
	close(jobs)

	wg.Wait()
	close(results)

	for res := range results {
		pop[res.index].Fitness = res.fitness
		pop[res.index].Evaluated = true
	}
}
