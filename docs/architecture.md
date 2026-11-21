# Architecture Specification

`github.com/Semplicementeio/Genetic-Algorithm` is an extensible, zero-dependency evolutionary optimization library written in Go 1.22+. It cleanly decouples generic evolutionary algorithms from domain-specific problem formulations.

---

## 1. Domain vs. Framework Separation

The repository strictly separates general-purpose genetic operators from specific problem domain models:

```
Genetic-Algorithm/
├── genetic/                   # Core Evolutionary Framework
│   ├── config.go              # Parametric options & validation
│   ├── individual.go          # Generic Individual[T] & Population[T]
│   ├── fitness.go             # Parallel fitness evaluator (panic/NaN safe)
│   ├── parallel.go            # Block scheduler, per-chunk random streams
│   ├── selection.go           # Tournament, Roulette Wheel, Rank, Elitism (+ PreparableSelector)
│   ├── crossover.go           # Single-Point, Two-Point, Uniform, Order (OX1)
│   ├── mutation.go            # BitFlip, Byte, Gaussian, Swap, Inversion
│   ├── statistics.go          # Generational statistics & CSV exporter
│   └── engine.go              # Evolutionary solver execution loop
│
├── examples/                  # Domain-Specific Problem Implementations
│   ├── knapsack/              # Binary bitset representation
│   ├── tsp/                   # Permutation genome & Order Crossover
│   ├── optimization/          # Continuous float64 math minimization
│   └── scheduling/            # Integer machine assignment representation
│
└── benchmarks/                # Performance & Scalability Benchmark Suite
```

---

## 2. Type-Safe Generic Abstractions (`[T any]`)

By leveraging Go 1.18+ generics, the library operates on arbitrary representations (`T`) without runtime reflection or interface boxing overhead:

### Core Interfaces

```go
type Individual[T any] struct {
    Genome    T
    Fitness   float64
    Evaluated bool
}

type FitnessFunc[T any] func(genome T) float64

type Selector[T any] interface {
    Select(pop Population[T], rng *rand.Rand) Individual[T]
}

type Crossover[T any] interface {
    Cross(p1, p2 T, rng *rand.Rand) (T, T)
}

type Mutator[T any] interface {
    Mutate(g T, rate float64, rng *rand.Rand) T
}
```

---

## 3. Concurrency Model: Block-Scheduled Parallel Execution

Both fitness evaluation and offspring creation are parallelized by one primitive, `runParallel(ctx, n, workers, grain, fn)`:

```
        [ Engine loop (one goroutine) ]
                    │
      builds a list of independent work items
   (unevaluated individuals / chunks of parent pairs)
                    │
         shared atomic counter: next block of `grain` items
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
    Worker 1    Worker 2 …  Worker N      (goroutines, one set per phase)
        │           │           │
        └── write only their own, distinct slice slots ──┘
```

- **No per-item synchronization**: workers claim blocks of items through a single atomic counter. An earlier design that sent every individual through a channel spent 30-40% of CPU on channel locking when fitness was cheap.
- **Parallel breeding**: the next generation is built in chunks of 32 parent pairs. The engine draws one seed per chunk from the main generator (sequentially), and each chunk uses its own private `rand.Rand` (a SplitMix64 source with O(1) seeding). Chunk boundaries do not depend on `NumWorkers`.
- **Errors and cancellation**: a panic inside any user function is recovered in the worker and returned as an error wrapping `ErrPanicRecovered`; a `NaN` fitness returns `ErrNaNFitness`; a canceled `context` stops all workers between blocks. The first error wins and remaining work is abandoned.

### Key Concurrency Invariants:
1. **Race-Free State**: Every work item writes only to its own slice element (`pop[i]` or its output slots), so no locks are needed; `go test -race` covers evaluation, breeding and the island model.
2. **Determinism Guarantee**: Results depend only on `Config.Seed`, never on `NumWorkers` or scheduling: random draws happen either on the main loop (chunk seeds, initial population) or inside a chunk's private stream. This is verified for 1-16 workers.
3. **Concurrency contract for user code**: with `NumWorkers > 1`, `FitnessFunc`, `GenomeCopier`, `Selector`, `Crossover` and `Mutator` are called concurrently and must not mutate shared state or inputs. `PreparableSelector.Prepare` is called once per generation on the main loop and returns a read-only snapshot.

---

## 4. Coarse-Grained Parallelism: Island Model GA (`IslandEngine[T]`)

In addition to the fine-grained parallelism above, `genetic` implements the **Island Model** (`IslandEngine[T]`) for coarse-grained parallel evolutionary computation:

```
    Island 1 (Goroutine)  ─── Migrants (Ring Topology) ───►  Island 2 (Goroutine)
             ▲                                                        │
             │                                                        ▼
    Island 4 (Goroutine)  ◄─── Migrants (Ring Topology) ───  Island 3 (Goroutine)
```

- **Independent Sub-Populations**: Spawns $M$ parallel islands evolving independently for $E$ epoch generations. Observers and control callbacks run serially at epoch boundaries on the merged population.
- **Migration Topologies**: Periodically exchanges elite migrants across islands (Ring, Random or Fully-Connected topologies), maintaining population diversity and preventing premature local optima convergence.

---

## 5. Error Handling & Parameter Validation

`Config[T].Validate()` returns explicit sentinel errors before execution starts:
- `ErrInvalidPopulationSize`
- `ErrInvalidMutationRate`
- `ErrNilFitnessFunc`
- `ErrNilGenerator`
- `ErrNilSelector`
- `ErrNilCrossover`
- `ErrNilMutator`

Runtime errors: `ErrPanicRecovered` (panic in a user function), `ErrNaNFitness`, and `ctx.Err()` on cancellation. `Run`/`RunContext` also report a typed `StopReason` and the effective `Seed`.

---

## 6. Design Decisions

- **Why Go generics (`[T any]`) over `interface{}`?** Compile-time type safety while the engine works on any genome representation, without reflection or type assertions.
- **Why block scheduling over goroutine-per-individual or a job channel?** Goroutine-per-individual causes GC and scheduler churn; a shared job channel makes workers contend on its lock when fitness is cheap. Each phase starts `NumWorkers` goroutines that claim blocks of work through one atomic counter.
- **Why fixed-size chunks with private random streams?** Parallel breeding must not make results depend on scheduling or worker count. Offspring are produced in chunks of constant size, each seeded in order from the main generator.
- **Why a ring topology as the island default?** It regulates information flow, slowing the spread of dominant individuals and helping preserve diversity across islands.
- **Why separate read-only observers from control hooks?** Logging should not be able to change the evolutionary state; keeping `Observers` apart from `OnGeneration` guarantees that.

## 7. When Not to Use a Genetic Algorithm

1. **Differentiable, well-behaved objectives**: gradient-based methods (gradient descent, L-BFGS) are usually far more efficient.
2. **Small discrete search spaces**: exhaustive search is simpler and guarantees the optimum.
3. **Island model on small problems**: islands add synchronization and migration overhead; for small populations or cheap fitness a single population can be faster.
