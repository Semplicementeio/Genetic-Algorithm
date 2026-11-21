package genetic

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"time"
)

// StopReason describes why an evolutionary run terminated.
type StopReason string

const (
	// StopMaxGenerations: the configured number of generations was completed.
	StopMaxGenerations StopReason = "max_generations_reached"
	// StopTargetFitness: Config.TargetFitness was reached.
	StopTargetFitness StopReason = "target_fitness_reached"
	// StopStalled: Config.MaxGenerationsWithoutProgress generations passed without improvement.
	StopStalled StopReason = "stalled"
	// StopUserCallback: Config.OnGeneration returned false.
	StopUserCallback StopReason = "user_callback_stop"
	// StopContextCanceled: the context passed to RunContext was canceled or timed out.
	StopContextCanceled StopReason = "context_canceled"
)

// Result contains the final output and metrics of an evolutionary optimization run.
type Result[T any] struct {
	BestIndividual       Individual[T]
	FinalPopulation      Population[T]
	GenerationsEvaluated int
	TotalDuration        time.Duration
	History              History

	// ConvergedReason is the machine-readable termination cause and StopDetail a
	// human-readable explanation (e.g. "12.0000 >= 10.0000").
	ConvergedReason StopReason
	StopDetail      string

	// Seed is the PRNG seed actually used. If Config.Seed was 0 (random), pass this
	// value as Config.Seed to reproduce the run.
	Seed int64
}

// breedChunkPairs is the number of parent pairs bred per random stream. It is a
// constant (not derived from NumWorkers) so results are identical for any worker count.
const breedChunkPairs = 32

// Engine drives the genetic algorithm execution loop.
//
// Run and RunContext can be called repeatedly (and concurrently): each call
// starts from Config.Seed, so the same configuration yields the same result.
type Engine[T any] struct {
	config    Config[T]
	seed      int64
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

	return &Engine[T]{
		config:    cfg,
		seed:      seed,
		evaluator: NewEvaluator(cfg.FitnessFunc, cfg.NumWorkers),
	}, nil
}

// Run executes the evolutionary optimization process.
func (e *Engine[T]) Run() (Result[T], error) {
	return e.RunContext(context.Background())
}

// RunContext executes the evolutionary optimization process until it finishes or
// ctx is done. On cancellation it returns the best result found so far (with
// ConvergedReason == StopContextCanceled) together with ctx.Err(). Any other
// error (e.g. a panic in the fitness function) is returned with an empty Result.
func (e *Engine[T]) RunContext(ctx context.Context) (Result[T], error) {
	rng := rand.New(rand.NewSource(e.seed))

	initialPop := make(Population[T], e.config.PopulationSize)
	for i := range initialPop {
		initialPop[i] = Individual[T]{Genome: e.config.Generator(rng)}
	}
	return e.runWithInitialPopulation(ctx, rng, initialPop)
}

func (e *Engine[T]) runWithInitialPopulation(ctx context.Context, rng *rand.Rand, initialPop Population[T]) (Result[T], error) {
	startTime := time.Now()
	pop := make(Population[T], len(initialPop))
	copy(pop, initialPop)

	// Initial fitness evaluation
	if err := e.evaluator.EvaluateContext(ctx, pop); err != nil {
		return Result[T]{}, fmt.Errorf("evaluating initial population: %w", err)
	}

	history := make(History, 0, e.config.Generations)
	bestOverall := pop.Best().Clone(e.config.GenomeCopier)
	generationsStalled := 0
	reason := StopMaxGenerations
	detail := ""
	var ctxErr error

	// genStart marks the beginning of the work that produced the current
	// population (breeding + evaluation; for generation 1, the initial
	// evaluation), so GenerationStats.Duration reflects the real cost of a
	// generation rather than just the time spent computing statistics.
	genStart := startTime

	for gen := 1; gen <= e.config.Generations; gen++ {
		best := pop.Best()
		avg := pop.AverageFitness()
		worst := pop.Worst()
		stdDev := CalculateStdDev(pop, avg)

		if best.Fitness > bestOverall.Fitness {
			bestOverall = best.Clone(e.config.GenomeCopier)
			generationsStalled = 0
		} else if gen > 1 {
			// The initial population is the baseline, not a stalled generation.
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
		stats := history[len(history)-1]

		// Invoke read-only Observers
		for _, obs := range e.config.Observers {
			if obs != nil {
				obs(stats, pop)
			}
		}

		// Custom control callback hook check
		if e.config.OnGeneration != nil && !e.config.OnGeneration(stats, pop) {
			reason = StopUserCallback
			break
		}

		// Target fitness check
		if e.config.TargetFitness != nil && best.Fitness >= *e.config.TargetFitness {
			reason = StopTargetFitness
			detail = fmt.Sprintf("%.4f >= %.4f", best.Fitness, *e.config.TargetFitness)
			break
		}

		// Stall detection check
		if e.config.MaxGenerationsWithoutProgress > 0 && generationsStalled >= e.config.MaxGenerationsWithoutProgress {
			reason = StopStalled
			detail = fmt.Sprintf("%d generations without improvement", generationsStalled)
			break
		}

		// If this is the last generation, stop before breeding a new population
		if gen == e.config.Generations {
			break
		}

		// Observer/callback time is excluded from the next generation's Duration.
		genStart = time.Now()

		nextPop, err := e.breed(ctx, rng, pop)
		if err == nil {
			err = e.evaluator.EvaluateContext(ctx, nextPop)
		}
		if err != nil {
			if ctx.Err() != nil && errors.Is(err, ctx.Err()) {
				// Keep the last fully evaluated generation as the final population.
				ctxErr = err
				break
			}
			return Result[T]{}, fmt.Errorf("generation %d: %w", gen+1, err)
		}
		pop = nextPop
	}

	if ctxErr != nil {
		reason = StopContextCanceled
		detail = ctxErr.Error()
	}

	return Result[T]{
		BestIndividual:       bestOverall,
		FinalPopulation:      pop,
		GenerationsEvaluated: len(history),
		TotalDuration:        time.Since(startTime),
		History:              history,
		ConvergedReason:      reason,
		StopDetail:           detail,
		Seed:                 e.seed,
	}, ctxErr
}

// breed builds the next (unevaluated) generation: elites first, then offspring.
//
// Offspring are produced in fixed-size chunks of parent pairs. Every chunk draws
// from its own random stream seeded (in order) from rng, and writes to distinct
// slots of the output slice, so chunks can run on any number of workers without
// changing the result.
func (e *Engine[T]) breed(ctx context.Context, rng *rand.Rand, pop Population[T]) (Population[T], error) {
	cfg := e.config
	size := cfg.PopulationSize
	next := make(Population[T], size)

	elites := 0
	if cfg.ElitismCount > 0 {
		elites = copy(next, ExtractElite(pop, cfg.ElitismCount, cfg.GenomeCopier))
	}

	pairs := (size - elites + 1) / 2
	chunks := (pairs + breedChunkPairs - 1) / breedChunkPairs
	seeds := make([]int64, chunks)
	for i := range seeds {
		seeds[i] = rng.Int63()
	}

	selector := prepareSelector(cfg.Selector, pop)

	err := runParallel(ctx, chunks, cfg.NumWorkers, 1, func(c int) error {
		crng := newChunkRand(seeds[c])
		last := (c + 1) * breedChunkPairs
		if last > pairs {
			last = pairs
		}
		for p := c * breedChunkPairs; p < last; p++ {
			slot := elites + 2*p
			p1 := selector.Select(pop, crng)
			p2 := selector.Select(pop, crng)

			var c1, c2 Individual[T]
			if crng.Float64() < cfg.CrossoverRate {
				g1, g2 := cfg.Crossover.Cross(p1.Genome, p2.Genome, crng)
				c1, c2 = Individual[T]{Genome: g1}, Individual[T]{Genome: g2}
			} else {
				// Clones keep their parent's fitness, which stays valid as long as
				// no mutation is applied below.
				c1, c2 = p1.Clone(cfg.GenomeCopier), p2.Clone(cfg.GenomeCopier)
			}

			if cfg.MutationRate > 0 {
				c1 = Individual[T]{Genome: cfg.Mutator.Mutate(c1.Genome, cfg.MutationRate, crng)}
				c2 = Individual[T]{Genome: cfg.Mutator.Mutate(c2.Genome, cfg.MutationRate, crng)}
			}

			next[slot] = c1
			if slot+1 < size {
				next[slot+1] = c2
			}
		}
		return nil
	})
	return next, err
}
