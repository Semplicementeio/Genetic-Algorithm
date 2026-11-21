package genetic

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"sort"
	"sync"
	"time"
)

var (
	ErrInvalidIslandCount    = errors.New("number of islands must be at least 2")
	ErrInvalidEpochs         = errors.New("epoch generations must be at least 1")
	ErrInvalidMigrationCount = errors.New("migration count must be at least 1 and less than the island population size")
	ErrInvalidTopology       = errors.New("unknown migration topology")
)

// Topology defines how migrants travel between islands.
type Topology int

const (
	// RingTopology sends the migrants of island i to island (i+1) mod N.
	RingTopology Topology = iota
	// RandomTopology sends the migrants of each island to one random other island.
	RandomTopology
	// FullyConnectedTopology sends the migrants of each island to every other island.
	// It requires MigrationCount*(NumIslands-1) < IslandPopulationSize.
	FullyConnectedTopology
)

// IslandConfig defines parameters for multi-population Island Model optimization.
//
// Semantics:
//   - Each island evolves independently for EpochGenerations generations (fewer
//     for the last epoch if TotalGenerations is not a multiple of it), then the
//     best MigrationCount individuals of every island migrate according to Topology.
//   - Island evolution runs concurrently, one goroutine per island, so
//     BaseConfig.FitnessFunc, Generator, Selector, Crossover and Mutator must be
//     safe for concurrent use even when BaseConfig.NumWorkers is 1.
//   - BaseConfig.Observers and BaseConfig.OnGeneration are invoked serially, once
//     per epoch, with aggregate statistics and the merged population of all
//     islands. TargetFitness and MaxGenerationsWithoutProgress are also evaluated
//     globally at epoch boundaries (stall is counted in generations).
//   - BaseConfig.PopulationSize and BaseConfig.Generations are ignored in favor of
//     IslandPopulationSize and the epoch settings.
type IslandConfig[T any] struct {
	NumIslands           int
	IslandPopulationSize int
	EpochGenerations     int
	MigrationCount       int
	TotalGenerations     int
	Topology             Topology
	BaseConfig           Config[T]
}

// Validate checks island configuration parameters.
func (ic *IslandConfig[T]) Validate() error {
	if ic.NumIslands < 2 {
		return fmt.Errorf("%w: got %d", ErrInvalidIslandCount, ic.NumIslands)
	}
	if ic.IslandPopulationSize < 2 {
		return fmt.Errorf("%w: got %d", ErrInvalidPopulationSize, ic.IslandPopulationSize)
	}
	if ic.EpochGenerations < 1 {
		return fmt.Errorf("%w: got %d", ErrInvalidEpochs, ic.EpochGenerations)
	}
	if ic.MigrationCount < 1 || ic.MigrationCount >= ic.IslandPopulationSize {
		return fmt.Errorf("%w: got %d for island pop size %d", ErrInvalidMigrationCount, ic.MigrationCount, ic.IslandPopulationSize)
	}
	if ic.TotalGenerations <= 0 {
		return fmt.Errorf("%w: got %d", ErrInvalidGenerations, ic.TotalGenerations)
	}

	switch ic.Topology {
	case RingTopology, RandomTopology:
	case FullyConnectedTopology:
		if incoming := ic.MigrationCount * (ic.NumIslands - 1); incoming >= ic.IslandPopulationSize {
			return fmt.Errorf("%w: fully connected topology delivers %d migrants per island but island pop size is %d",
				ErrInvalidMigrationCount, incoming, ic.IslandPopulationSize)
		}
	default:
		return fmt.Errorf("%w: %d", ErrInvalidTopology, ic.Topology)
	}

	cfgCopy := ic.BaseConfig
	cfgCopy.PopulationSize = ic.IslandPopulationSize
	cfgCopy.Generations = ic.EpochGenerations
	return cfgCopy.Validate()
}

// IslandEngine implements parallel island-model evolutionary optimization.
//
// Like Engine, Run and RunContext can be called repeatedly: every call starts
// from the configured seed and yields the same result.
type IslandEngine[T any] struct {
	config IslandConfig[T]
	seed   int64
}

// NewIslandEngine creates a validated IslandEngine instance.
func NewIslandEngine[T any](cfg IslandConfig[T]) (*IslandEngine[T], error) {
	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("invalid island configuration: %w", err)
	}

	seed := cfg.BaseConfig.Seed
	if seed == 0 {
		seed = time.Now().UnixNano()
	}

	return &IslandEngine[T]{config: cfg, seed: seed}, nil
}

// Run executes coarse-grained parallel island evolution with periodic migration.
func (ie *IslandEngine[T]) Run() (Result[T], error) {
	return ie.RunContext(context.Background())
}

// RunContext is like Run but stops when ctx is done. On cancellation it returns
// the best result found so far (ConvergedReason == StopContextCanceled) together
// with ctx.Err(); any other error is returned with an empty Result.
func (ie *IslandEngine[T]) RunContext(ctx context.Context) (Result[T], error) {
	startTime := time.Now()
	rng := rand.New(rand.NewSource(ie.seed))
	cfg := ie.config
	base := cfg.BaseConfig
	numIslands := cfg.NumIslands

	// Initialize island populations
	islands := make([]Population[T], numIslands)
	for i := 0; i < numIslands; i++ {
		islands[i] = make(Population[T], cfg.IslandPopulationSize)
		islandRNG := rand.New(rand.NewSource(rng.Int63()))
		for j := 0; j < cfg.IslandPopulationSize; j++ {
			islands[i][j] = Individual[T]{Genome: base.Generator(islandRNG)}
		}
	}

	evaluator := NewEvaluator(base.FitnessFunc, base.NumWorkers)
	for i := 0; i < numIslands; i++ {
		if err := evaluator.EvaluateContext(ctx, islands[i]); err != nil {
			return Result[T]{}, fmt.Errorf("evaluating initial population of island %d: %w", i, err)
		}
	}

	history := make(History, 0, (cfg.TotalGenerations+cfg.EpochGenerations-1)/cfg.EpochGenerations)
	bestOverall := ie.mergeIslands(islands).Best().Clone(base.GenomeCopier)
	generationsStalled := 0
	reason := StopMaxGenerations
	detail := ""
	var ctxErr error
	currentGen := 0

	for currentGen < cfg.TotalGenerations {
		epochGens := cfg.EpochGenerations
		if remaining := cfg.TotalGenerations - currentGen; remaining < epochGens {
			epochGens = remaining
		}

		// Draw per-island seeds serially so runs stay reproducible.
		seeds := make([]int64, numIslands)
		for i := range seeds {
			seeds[i] = rng.Int63()
			if seeds[i] == 0 {
				seeds[i] = 1 // 0 means "pick a random seed" for Engine
			}
		}

		epochStart := time.Now()
		errs := make([]error, numIslands)
		var wg sync.WaitGroup
		wg.Add(numIslands)

		for islandIdx := 0; islandIdx < numIslands; islandIdx++ {
			go func(idx int) {
				defer wg.Done()

				islandCfg := base
				islandCfg.PopulationSize = cfg.IslandPopulationSize
				// The starting population counts as generation 1 of the inner run,
				// so epochGens+1 yields exactly epochGens rounds of breeding.
				islandCfg.Generations = epochGens + 1
				islandCfg.Seed = seeds[idx]
				// Run-level control is handled globally at epoch boundaries.
				islandCfg.Observers = nil
				islandCfg.OnGeneration = nil
				islandCfg.TargetFitness = nil
				islandCfg.MaxGenerationsWithoutProgress = 0

				eng, err := NewEngine(islandCfg)
				if err != nil {
					errs[idx] = err
					return
				}
				// Each island gets its own random stream, independent of scheduling.
				res, err := eng.runWithInitialPopulation(ctx, rand.New(rand.NewSource(seeds[idx])), islands[idx])
				if res.FinalPopulation != nil {
					islands[idx] = res.FinalPopulation
				}
				errs[idx] = err
			}(islandIdx)
		}

		wg.Wait()
		for idx, err := range errs {
			if err == nil {
				continue
			}
			if ctx.Err() != nil && errors.Is(err, ctx.Err()) {
				ctxErr = err
				continue
			}
			return Result[T]{}, fmt.Errorf("island %d failed at generation %d: %w", idx, currentGen, err)
		}
		if ctxErr != nil {
			// Islands hold their last fully evaluated generation; keep the best seen.
			if best := ie.mergeIslands(islands).Best(); best.Fitness > bestOverall.Fitness {
				bestOverall = best.Clone(base.GenomeCopier)
			}
			reason = StopContextCanceled
			detail = ctxErr.Error()
			break
		}
		currentGen += epochGens

		// Aggregate statistics across all islands for this epoch
		merged := ie.mergeIslands(islands)
		best := merged.Best()
		avg := merged.AverageFitness()
		if best.Fitness > bestOverall.Fitness {
			bestOverall = best.Clone(base.GenomeCopier)
			generationsStalled = 0
		} else {
			generationsStalled += epochGens
		}

		stats := GenerationStats{
			Generation:   currentGen,
			BestFitness:  best.Fitness,
			AvgFitness:   avg,
			WorstFitness: merged.Worst().Fitness,
			StdDev:       CalculateStdDev(merged, avg),
			Duration:     time.Since(epochStart),
		}
		history.Record(stats)

		for _, obs := range base.Observers {
			if obs != nil {
				obs(stats, merged)
			}
		}

		if base.OnGeneration != nil && !base.OnGeneration(stats, merged) {
			reason = StopUserCallback
			break
		}

		if base.TargetFitness != nil && best.Fitness >= *base.TargetFitness {
			reason = StopTargetFitness
			detail = fmt.Sprintf("%.4f >= %.4f", best.Fitness, *base.TargetFitness)
			break
		}

		if base.MaxGenerationsWithoutProgress > 0 && generationsStalled >= base.MaxGenerationsWithoutProgress {
			reason = StopStalled
			detail = fmt.Sprintf("%d generations without improvement", generationsStalled)
			break
		}

		if currentGen >= cfg.TotalGenerations {
			break
		}

		// Perform migration between islands
		ie.migrate(islands, rng)
	}

	return Result[T]{
		BestIndividual:       bestOverall,
		FinalPopulation:      ie.mergeIslands(islands),
		GenerationsEvaluated: currentGen,
		TotalDuration:        time.Since(startTime),
		History:              history,
		ConvergedReason:      reason,
		StopDetail:           detail,
		Seed:                 ie.seed,
	}, ctxErr
}

// mergeIslands returns a flat snapshot of all island populations. The returned
// slice is new, but genomes are shared with the islands.
func (ie *IslandEngine[T]) mergeIslands(islands []Population[T]) Population[T] {
	merged := make(Population[T], 0, len(islands)*ie.config.IslandPopulationSize)
	for _, isl := range islands {
		merged = append(merged, isl...)
	}
	return merged
}

// targets returns the destination islands for the migrants of island src.
func (ie *IslandEngine[T]) targets(src, numIslands int, rng *rand.Rand) []int {
	switch ie.config.Topology {
	case RandomTopology:
		t := rng.Intn(numIslands - 1)
		if t >= src {
			t++
		}
		return []int{t}
	case FullyConnectedTopology:
		all := make([]int, 0, numIslands-1)
		for i := 0; i < numIslands; i++ {
			if i != src {
				all = append(all, i)
			}
		}
		return all
	default:
		return []int{(src + 1) % numIslands}
	}
}

func (ie *IslandEngine[T]) migrate(islands []Population[T], rng *rand.Rand) {
	numIslands := len(islands)
	copier := ie.config.BaseConfig.GenomeCopier

	// Extract top migrants from every island before any island is modified.
	migrants := make([]Population[T], numIslands)
	for i := 0; i < numIslands; i++ {
		migrants[i] = ExtractElite(islands[i], ie.config.MigrationCount, copier)
	}

	incoming := make([]Population[T], numIslands)
	for i := 0; i < numIslands; i++ {
		for _, t := range ie.targets(i, numIslands, rng) {
			incoming[t] = append(incoming[t], migrants[i]...)
		}
	}

	for t, arrivals := range incoming {
		pop := islands[t]
		// Always keep at least one native individual.
		k := len(arrivals)
		if k > len(pop)-1 {
			k = len(pop) - 1
		}
		if k == 0 {
			continue
		}

		// Keep the best arrivals if more migrants than slots are available.
		sort.SliceStable(arrivals, func(a, b int) bool { return arrivals[a].Fitness > arrivals[b].Fitness })

		// Replace the k worst natives (determined before any replacement, so a
		// migrant can never overwrite another migrant).
		order := make([]int, len(pop))
		for i := range order {
			order[i] = i
		}
		sort.SliceStable(order, func(a, b int) bool { return pop[order[a]].Fitness < pop[order[b]].Fitness })

		for i := 0; i < k; i++ {
			pop[order[i]] = arrivals[i].Clone(copier)
		}
	}
}
