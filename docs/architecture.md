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
│   ├── fitness.go             # Parallel worker pool evaluator
│   ├── selection.go           # Tournament, Roulette Wheel, Rank, Elitism
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

## 3. Concurrency Model: Parallel Worker Pool

Fitness evaluation is frequently the primary computational bottleneck in evolutionary algorithms. The library implements a lock-free, channel-based worker pool pattern:

```
          [ Main Engine Loop ]
                   │
         Dispatches Unevaluated
             Genomes (Jobs)
                   │
           ┌───────┴───────┐
           ▼               ▼
     Worker 1        Worker 2    ...  Worker N
   (Goroutine)     (Goroutine)      (Goroutine)
           │               │               │
           └───────┬───────┘               │
                   ▼                       │
         Collects Results Channel ◄────────┘
                   │
          Updates Population
```

### Key Concurrency Invariants:
1. **Race-Free State**: Goroutines calculate pure functions `FitnessFunc(Genome)` and write results through buffered channels back to the main thread.
2. **Determinism Guarantee**: Random Number Generator (`rand.Rand`) calls occur exclusively on the single-threaded main loop. Worker threads perform zero PRNG mutations, guaranteeing 100% bit-identical reproducibility regardless of worker count.

---

## 4. Coarse-Grained Parallelism: Island Model GA (`IslandEngine[T]`)

In addition to worker-pool fine-grained parallelism, `genetic` implements the **Island Model** (`IslandEngine[T]`) for coarse-grained parallel evolutionary computation:

```
    Island 1 (Goroutine)  ─── Migrants (Ring Topology) ───►  Island 2 (Goroutine)
             ▲                                                        │
             │                                                        ▼
    Island 4 (Goroutine)  ◄─── Migrants (Ring Topology) ───  Island 3 (Goroutine)
```

- **Independent Sub-Populations**: Spawns $M$ parallel islands evolving independently for $E$ epoch generations.
- **Migration Topologies**: Periodically exchanges elite migrants across islands (Ring or Random topologies), maintaining population diversity and preventing premature local optima convergence.

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
