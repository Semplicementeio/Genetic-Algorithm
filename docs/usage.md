# Usage Guide

Practical reference for the features of the `genetic` package beyond the basic example in the [README](../README.md). For the internals see [architecture.md](architecture.md); for measurements see [experiments.md](experiments.md).

Contents: [Choosing operators](#1-choosing-operators) · [Progress and early stopping](#2-progress-monitoring-and-early-stopping) · [Reproducibility](#3-reproducibility-and-parallelism) · [Cancellation and errors](#4-cancellation-errors-and-stop-reasons) · [Island model](#5-island-model) · [Custom selectors](#6-custom-selectors) · [CSV export](#7-exporting-results)

---

## 1. Choosing operators

Every run needs a **selector** (who reproduces), a **crossover** (how two parents are combined) and a **mutator** (small random changes). Pick them according to how a candidate solution (the *genome*) is represented.

| Category | Operator | Genome type | Typical use |
|----------|----------|-------------|-------------|
| **Selection** | `TournamentSelection[T]` | any | Default choice; the tournament size `K` sets how strongly the best are favored |
| | `RouletteWheelSelection[T]` | any | Probability proportional to fitness |
| | `RankSelection[T]` | any | Probability proportional to rank; robust when fitness values vary wildly |
| | `ExtractElite[T]` | any | Elitism helper: the top *N* individuals |
| **Crossover** | `SinglePointCrossover` | slices | Generic, positional |
| | `TwoPointCrossover` | slices | Keeps contiguous blocks of genes together |
| | `UniformCrossover` | slices | Gene-by-gene mixing |
| | `OrderCrossover` (OX1) | slices of distinct values | **Permutations** (routes, orderings); never creates duplicates |
| **Mutation** | `BitFlipMutation` | `[]bool` | Yes/no choices |
| | `ByteMutation` | `[]byte` | Text / small alphabets |
| | `GaussianMutation` | `[]float64` | Continuous parameters (adds normally distributed noise) |
| | `SwapMutation` | any slice | Permutations: swaps two positions |
| | `InversionMutation` | any slice | Permutations: reverses a segment |

Note on `MutationRate`: for `BitFlip`, `Byte` and `Gaussian` it is the probability of changing **each gene**; for `Swap` and `Inversion` it is the probability of mutating **the whole genome once**.

You can implement your own: `Selector[T]`, `Crossover[T]` and `Mutator[T]` are small interfaces.

---

## 2. Progress monitoring and early stopping

`Observers` only watch; `OnGeneration` can stop the run.

```go
// Read-only: logging, progress bars, plots
cfg.Observers = []func(genetic.GenerationStats, genetic.Population[T]){
	func(s genetic.GenerationStats, pop genetic.Population[T]) {
		fmt.Printf("Gen %d | Best: %.4f | Avg: %.4f | StdDev: %.4f\n",
			s.Generation, s.BestFitness, s.AvgFitness, s.StdDev)
	},
}

// Control: return false to stop
cfg.OnGeneration = func(s genetic.GenerationStats, pop genetic.Population[T]) bool {
	return s.BestFitness < 100.0 && s.StdDev > 0.0001 // stop on target or when diversity collapses
}
```

Built-in stop conditions: `Config.TargetFitness` (stop when reached) and `Config.MaxGenerationsWithoutProgress` (stop after that many generations without a new best).

---

## 3. Reproducibility and parallelism

- A fixed `Seed` gives the **same result on every run and for every `NumWorkers`**. Calling `Run()` again on the same engine replays the same run.
- `Seed: 0` picks a random seed. The seed actually used is returned in `Result.Seed`; pass it back as `Config.Seed` to replay that run exactly.
- `NumWorkers` sets how many CPU threads evaluate fitness *and* create offspring.

**Concurrency contract.** With `NumWorkers > 1`, your `FitnessFunc`, `GenomeCopier`, `Selector`, `Crossover` and `Mutator` are called from several goroutines at once. They must not modify shared state or their inputs (all built-in operators comply). The same applies to the island model even with `NumWorkers: 1`, because islands run concurrently.

---

## 4. Cancellation, errors and stop reasons

```go
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()

res, err := engine.RunContext(ctx) // IslandEngine has RunContext too
switch {
case errors.Is(err, context.DeadlineExceeded):
	// res holds the best individual found so far (res.ConvergedReason == genetic.StopContextCanceled)
case errors.Is(err, genetic.ErrPanicRecovered), errors.Is(err, genetic.ErrNaNFitness):
	// a user function panicked or returned NaN: reported as an error, not a crash
}

switch res.ConvergedReason { // StopMaxGenerations, StopTargetFitness, StopStalled, StopUserCallback, StopContextCanceled
case genetic.StopTargetFitness:
	fmt.Println(res.StopDetail) // e.g. "31.0000 >= 31.0000"
}
```

---

## 5. Island model

`IslandEngine[T]` evolves several independent sub-populations ("islands") in parallel. Every `EpochGenerations` generations, the best individuals of each island migrate to others. This helps preserve diversity and avoid getting stuck in a poor local optimum.

```go
baseCfg := genetic.Config[[]float64]{ /* same fields as a normal run */ }

islandCfg := genetic.IslandConfig[[]float64]{
	NumIslands:           4,                  // 4 parallel islands
	IslandPopulationSize: 50,                 // 50 individuals per island
	EpochGenerations:     10,                 // migrate every 10 generations
	MigrationCount:       2,                  // 2 individuals leave each island per migration
	TotalGenerations:     100,
	Topology:             genetic.RingTopology, // or RandomTopology, FullyConnectedTopology
	BaseConfig:           baseCfg,
}

eng, err := genetic.NewIslandEngine(islandCfg)
if err != nil { panic(err) }
res, err := eng.Run()
```

- `Observers`, `OnGeneration`, `TargetFitness` and `MaxGenerationsWithoutProgress` are applied once per epoch on the merged population of all islands; `History` has one entry per epoch.
- `FullyConnectedTopology` requires `MigrationCount × (NumIslands − 1) < IslandPopulationSize`.

---

## 6. Custom selectors

A selector whose work depends on the whole population (cumulative weights, rankings…) can implement `PreparableSelector[T]` (`Prepare(pop) Selector[T]`). The engine then calls `Prepare` once per generation instead of paying the cost on every `Select`. `RouletteWheelSelection` and `RankSelection` do this.

---

## 7. Exporting results

```go
f, _ := os.Create("evolution.csv")
defer f.Close()
res.History.ExportCSV(f) // generation, best/avg/worst fitness, std dev, duration per generation
```
