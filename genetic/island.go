package genetic

import (
	"errors"
	"fmt"
	"math/rand"
	"sync"
	"time"
)

var (
	ErrInvalidIslandCount     = errors.New("number of islands must be at least 2")
	ErrInvalidEpochs          = errors.New("epoch generations must be at least 1")
	ErrInvalidMigrationCount  = errors.New("migration count must be between 1 and island population size")
)

type Topology int

const (
	RingTopology Topology = iota
	RandomTopology
	FullyConnectedTopology
)

// IslandConfig defines parameters for multi-population Island Model optimization.
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

	cfgCopy := ic.BaseConfig
	cfgCopy.PopulationSize = ic.IslandPopulationSize
	cfgCopy.Generations = ic.EpochGenerations
	return cfgCopy.Validate()
}

// IslandEngine implements parallel island-model evolutionary optimization.
type IslandEngine[T any] struct {
	config IslandConfig[T]
	rng    *rand.Rand
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

	return &IslandEngine[T]{
		config: cfg,
		rng:    rand.New(rand.NewSource(seed)),
	}, nil
}

// Run executes coarse-grained parallel island evolution with periodic migration.
func (ie *IslandEngine[T]) Run() (Result[T], error) {
	startTime := time.Now()
	numIslands := ie.config.NumIslands
	epochs := ie.config.TotalGenerations / ie.config.EpochGenerations
	if epochs < 1 {
		epochs = 1
	}

	// Initialize island populations
	islands := make([]Population[T], numIslands)
	for i := 0; i < numIslands; i++ {
		islands[i] = make(Population[T], ie.config.IslandPopulationSize)
		islandRNG := rand.New(rand.NewSource(ie.rng.Int63() + int64(i)))
		for j := 0; j < ie.config.IslandPopulationSize; j++ {
			islands[i][j] = Individual[T]{
				Genome:    ie.config.BaseConfig.Generator(islandRNG),
				Evaluated: false,
			}
		}
	}

	evaluator := NewEvaluator(ie.config.BaseConfig.FitnessFunc, ie.config.BaseConfig.NumWorkers)
	for i := 0; i < numIslands; i++ {
		evaluator.Evaluate(islands[i])
	}

	history := make(History, 0, ie.config.TotalGenerations)
	var bestOverall Individual[T]
	bestOverallSet := false
	convergedReason := "max_generations_reached"

	currentGen := 0

	for epoch := 1; epoch <= epochs; epoch++ {
		// Run epoch for each island concurrently
		var wg sync.WaitGroup
		wg.Add(numIslands)

		for islandIdx := 0; islandIdx < numIslands; islandIdx++ {
			idx := islandIdx
			go func() {
				defer wg.Done()

				islandCfg := ie.config.BaseConfig
				islandCfg.PopulationSize = ie.config.IslandPopulationSize
				islandCfg.Generations = ie.config.EpochGenerations
				islandCfg.Seed = ie.config.BaseConfig.Seed + int64(epoch*100000+idx*1000)

				eng, err := NewEngine(islandCfg)
				if err != nil {
					return
				}
				// Inject starting population
				res, err := eng.runWithInitialPopulation(islands[idx])
				if err == nil {
					islands[idx] = res.FinalPopulation
				}
			}()
		}

		wg.Wait()
		currentGen += ie.config.EpochGenerations

		// Aggregate statistics across all islands for this epoch
		var globalBest Individual[T]
		var totalFitness float64
		totalInds := 0

		for _, isl := range islands {
			islBest := isl.Best()
			if !bestOverallSet || islBest.Fitness > bestOverall.Fitness {
				bestOverall = islBest.Clone(ie.config.BaseConfig.GenomeCopier)
				bestOverallSet = true
			}
			if totalInds == 0 || islBest.Fitness > globalBest.Fitness {
				globalBest = islBest
			}
			for _, ind := range isl {
				totalFitness += ind.Fitness
				totalInds++
			}
		}

		avgFit := totalFitness / float64(totalInds)
		history.Record(GenerationStats{
			Generation:  currentGen,
			BestFitness: globalBest.Fitness,
			AvgFitness:  avgFit,
			Duration:    time.Since(startTime),
		})

		// Check target fitness stop
		if ie.config.BaseConfig.TargetFitness != nil && globalBest.Fitness >= *ie.config.BaseConfig.TargetFitness {
			convergedReason = fmt.Sprintf("target_fitness_reached (%.4f >= %.4f)", globalBest.Fitness, *ie.config.BaseConfig.TargetFitness)
			break
		}

		// Perform migration between islands
		ie.migrate(islands)
	}

	// Merge all island populations into final population
	finalPop := make(Population[T], 0, numIslands*ie.config.IslandPopulationSize)
	for _, isl := range islands {
		finalPop = append(finalPop, isl...)
	}

	return Result[T]{
		BestIndividual:       bestOverall,
		FinalPopulation:      finalPop,
		GenerationsEvaluated: currentGen,
		TotalDuration:        time.Since(startTime),
		History:              history,
		ConvergedReason:      convergedReason,
	}, nil
}

func (ie *IslandEngine[T]) migrate(islands []Population[T]) {
	numIslands := len(islands)
	migrants := make([]Population[T], numIslands)

	// Extract top migrants from each island
	for i := 0; i < numIslands; i++ {
		migrants[i] = ExtractElite(islands[i], ie.config.MigrationCount, ie.config.BaseConfig.GenomeCopier)
	}

	// Transfer migrants according to topology
	for i := 0; i < numIslands; i++ {
		targetIdx := (i + 1) % numIslands // Ring topology
		if ie.config.Topology == RandomTopology {
			targetIdx = ie.rng.Intn(numIslands)
			for targetIdx == i {
				targetIdx = ie.rng.Intn(numIslands)
			}
		}

		// Replace worst individuals in target island with incoming migrants
		targetPop := islands[targetIdx]
		for mIdx, migrant := range migrants[i] {
			worstIdx := 0
			worstFit := targetPop[0].Fitness
			for j := 1; j < len(targetPop); j++ {
				if targetPop[j].Fitness < worstFit {
					worstFit = targetPop[j].Fitness
					worstIdx = j
				}
			}
			targetPop[worstIdx] = migrant.Clone(ie.config.BaseConfig.GenomeCopier)
			mIdx++
		}
	}
}
