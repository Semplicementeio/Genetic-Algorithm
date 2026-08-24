package genetic

import (
	"fmt"
	"math/rand"
	"time"
)

// Result contains the final output and metrics of an evolutionary optimization run.
type Result[T any] struct {
	BestIndividual       Individual[T]
	FinalPopulation      Population[T]
	GenerationsEvaluated int
	TotalDuration        time.Duration
	History              History
	ConvergedReason      string
}

// Engine drives the genetic algorithm execution loop.
type Engine[T any] struct {
	config    Config[T]
	rng       *rand.Rand
	evaluator *Evaluator[T]
}

// NewEngine creates and validates a new Engine instance.
func NewEngine[T any](cfg Config[T]) (*Engine[T], error) {
	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}

	seed := cfg.Seed
	if seed == 0 {
		seed = time.Now().UnixNano()
	}
	rng := rand.New(rand.NewSource(seed))

	evaluator := NewEvaluator(cfg.FitnessFunc, cfg.NumWorkers)

	return &Engine[T]{
		config:    cfg,
		rng:       rng,
		evaluator: evaluator,
	}, nil
}

// Run executes the evolutionary optimization process.
func (e *Engine[T]) Run() (Result[T], error) {
	initialPop := make(Population[T], e.config.PopulationSize)
	for i := 0; i < e.config.PopulationSize; i++ {
		initialPop[i] = Individual[T]{
			Genome:    e.config.Generator(e.rng),
			Evaluated: false,
		}
	}
	return e.runWithInitialPopulation(initialPop)
}

func (e *Engine[T]) runWithInitialPopulation(initialPop Population[T]) (Result[T], error) {
	startTime := time.Now()
	pop := make(Population[T], len(initialPop))
	copy(pop, initialPop)

	// 2. Initial fitness evaluation
	e.evaluator.Evaluate(pop)

	history := make(History, 0, e.config.Generations)
	bestOverall := pop.Best()
	generationsStalled := 0
	convergedReason := "max_generations_reached"

	// 3. Generational loop
	for gen := 1; gen <= e.config.Generations; gen++ {
		genStart := time.Now()

		best := pop.Best()
		avg := pop.AverageFitness()
		worst := pop.Worst()
		stdDev := CalculateStdDev(pop, avg)

		if best.Fitness > bestOverall.Fitness {
			bestOverall = best.Clone(e.config.GenomeCopier)
			generationsStalled = 0
		} else {
			generationsStalled++
		}

		history.Record(GenerationStats{
			Generation:   gen,
			BestFitness:  best.Fitness,
			AvgFitness:   avg,
			WorstFitness: worst.Fitness,
			StdDev:       stdDev,
			Duration:     time.Since(genStart),
		})

		// Invoke read-only Observers
		for _, obs := range e.config.Observers {
			if obs != nil {
				obs(history[len(history)-1], pop)
			}
		}

		// Custom control callback hook check
		if e.config.OnGeneration != nil {
			if !e.config.OnGeneration(history[len(history)-1], pop) {
				convergedReason = "user_callback_stop"
				break
			}
		}

		// Target fitness check
		if e.config.TargetFitness != nil && best.Fitness >= *e.config.TargetFitness {
			convergedReason = fmt.Sprintf("target_fitness_reached (%.4f >= %.4f)", best.Fitness, *e.config.TargetFitness)
			break
		}

		// Stall detection check
		if e.config.MaxGenerationsWithoutProgress > 0 && generationsStalled >= e.config.MaxGenerationsWithoutProgress {
			convergedReason = fmt.Sprintf("stalled_for_%d_generations", generationsStalled)
			break
		}

		// If this is the last generation, break before breeding new population
		if gen == e.config.Generations {
			break
		}

		// 4. Create next generation
		nextPop := make(Population[T], 0, e.config.PopulationSize)

		// Elitism: carry over top N individuals
		if e.config.ElitismCount > 0 {
			elites := ExtractElite(pop, e.config.ElitismCount, e.config.GenomeCopier)
			nextPop = append(nextPop, elites...)
		}

		// Reproduction loop
		for len(nextPop) < e.config.PopulationSize {
			p1 := e.config.Selector.Select(pop, e.rng)
			p2 := e.config.Selector.Select(pop, e.rng)

			var c1Genome, c2Genome T

			if e.rng.Float64() < e.config.CrossoverRate {
				c1Genome, c2Genome = e.config.Crossover.Cross(p1.Genome, p2.Genome, e.rng)
			} else {
				c1Genome = p1.Clone(e.config.GenomeCopier).Genome
				c2Genome = p2.Clone(e.config.GenomeCopier).Genome
			}

			// Mutation
			if e.config.MutationRate > 0 {
				c1Genome = e.config.Mutator.Mutate(c1Genome, e.config.MutationRate, e.rng)
				c2Genome = e.config.Mutator.Mutate(c2Genome, e.config.MutationRate, e.rng)
			}

			nextPop = append(nextPop, Individual[T]{Genome: c1Genome, Evaluated: false})
			if len(nextPop) < e.config.PopulationSize {
				nextPop = append(nextPop, Individual[T]{Genome: c2Genome, Evaluated: false})
			}
		}

		// Evaluate new population
		e.evaluator.Evaluate(nextPop)
		pop = nextPop
	}

	totalDuration := time.Since(startTime)
	finalBest := pop.Best()
	if finalBest.Fitness > bestOverall.Fitness {
		bestOverall = finalBest.Clone(e.config.GenomeCopier)
	}

	return Result[T]{
		BestIndividual:       bestOverall,
		FinalPopulation:      pop,
		GenerationsEvaluated: len(history),
		TotalDuration:        totalDuration,
		History:              history,
		ConvergedReason:      convergedReason,
	}, nil
}
