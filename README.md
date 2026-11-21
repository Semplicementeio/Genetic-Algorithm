# 🧬 Genetic-Algorithm

**Find good solutions to hard optimization problems by simulating evolution.**
A fast, reproducible genetic-algorithm library for Go, with runnable examples that need no programming.

[![CI](https://github.com/Semplicementeio/Genetic-Algorithm/actions/workflows/ci.yml/badge.svg)](https://github.com/Semplicementeio/Genetic-Algorithm/actions)
[![Go Reference](https://pkg.go.dev/badge/github.com/Semplicementeio/Genetic-Algorithm.svg)](https://pkg.go.dev/github.com/Semplicementeio/Genetic-Algorithm)
[![Go Report Card](https://goreportcard.com/badge/github.com/Semplicementeio/Genetic-Algorithm)](https://goreportcard.com/report/github.com/Semplicementeio/Genetic-Algorithm)

> **Status**: `v0.2.0`, in active development. The API may still change before `v1.0.0`.

> [!NOTE]
> **A note from the author, and a request for your feedback.**
> I am a software developer, not a scientist. I built this project while learning about evolutionary computation, so I cannot be certain that everything in it is correct: the algorithms, the way I describe them, the design of the experiments, or the conclusions I draw from the results.
>
> If you work in science or research, I would be very grateful for your opinion, even on small things: imprecise terminology, flaws in the methodology, missing references, features that would make this useful for your work, or better examples. I am open to every comment, criticism and suggestion.
>
> 📧 **Luca@semplicemente.io**

---

## What is this?

Some problems have far too many possible answers to try them all. A salesperson who must visit 20 cities can choose among about **60 quadrillion** (6 × 10¹⁶) different routes. Scheduling jobs on machines, choosing what to pack, or tuning the parameters of a model explode in the same way.

A **genetic algorithm** attacks such problems the way nature does:

1. **Start** with a population of random candidate answers (mostly terrible ones).
2. **Score** every candidate: how short is this route? how well does this parameter set fit the data?
3. **Reproduce** the best: pick good candidates as parents, combine their features (*crossover*) and apply small random changes (*mutation*).
4. **Repeat** for many generations. The population steadily improves.

This project is a ready-made, tested engine that runs that loop. **You describe your problem** (what a candidate looks like, and how to score it). **The engine does the evolution**, using all CPU cores, and returns the best solution found, how the search progressed, and why it stopped.

It is a software library written in [Go](https://go.dev). To apply it to a new problem someone writes a short program (about 30 lines, see [Use it in your own project](#use-it-in-your-own-project)). To see it work you need no programming at all, see [Try it](#try-it-no-programming-needed).

**Vocabulary**, using the route example:

| Term | Meaning | Route example |
|------|---------|---------------|
| Candidate (*genome*) | One possible answer | An ordering of the 20 cities |
| Fitness | Its score, higher is better | A number that grows as the route gets shorter |
| Population | The set of candidates | 200 routes |
| Generation | One round of scoring and reproduction | |
| Crossover | Combine two parents into children | Keep a stretch of route from A, fill the rest in B's order |
| Mutation | A small random change | Reverse a stretch of the route |
| Elitism | The very best are copied unchanged | The 4 best routes always survive |

---

## A concrete example

**Problem:** shortest closed tour through 20 Italian cities (the `tsp` example).

| Method | Route length | Candidates examined |
|--------|-------------:|--------------------:|
| Random route (average of 100,000) | 91.3 | – |
| Best of 100,000 random routes | 49.1 | 100,000 |
| **This library** (200 candidates × 300 generations) | **29.90** | **~60,000** |
| Exact optimum (verified with a full dynamic-programming computation) | 29.90 | all 6 × 10¹⁶ |

The evolved route has exactly the optimal length, found in about 50 milliseconds on a laptop, after examining a vanishing fraction of all possibilities. Lengths are in coordinate units (degrees of latitude/longitude).

Note that a genetic algorithm is a *heuristic*: it **does not guarantee** the optimum. Here it found it; on harder problems it returns a very good answer, not a proven best one. The other examples:

- **Job scheduling:** split 10 tasks over 3 machines so that all finish as early as possible. Result: 165 minutes on every machine, which equals the theoretical lower bound (495 minutes of work ÷ 3 machines).
- **Knapsack:** choose which of 12 items to pack under a 6 kg limit to maximize total value.
- **Mathematical test functions:** minimize two classic benchmark functions, compared against two other methods (see [Results](#how-well-does-it-work)).

---

## Try it (no programming needed)

1. Install Go 1.22 or newer from [go.dev/dl](https://go.dev/dl/).
2. Download this repository and, in its folder, run:

```bash
go run . -example tsp          # shortest route through 20 cities
go run . -example scheduling   # distribute tasks over 3 machines
go run . -example knapsack     # choose items under a weight limit
go run . -example optimization # compare the method with two alternatives (about a second)
go run . -seed 7               # run everything with another random seed
```

Results are reproducible: the same `-seed` always gives the same output.

---

## Use it in your own project

```bash
go get github.com/Semplicementeio/Genetic-Algorithm
```

The example below evolves a sentence starting from random letters. Four things define a problem:

```go
package main

import (
	"fmt"
	"math/rand"

	"github.com/Semplicementeio/Genetic-Algorithm/genetic"
)

func main() {
	target := []byte("EVOLUTIONARY OPTIMIZATION IN GO")
	alphabet := "ABCDEFGHIJKLMNOPQRSTUVWXYZ "

	cfg := genetic.Config[[]byte]{
		// Settings of the evolution
		PopulationSize: 150,  // candidates per generation
		Generations:    200,
		MutationRate:   0.04, // chance of a random change
		CrossoverRate:  0.85, // chance two parents are combined
		ElitismCount:   3,    // best candidates kept unchanged
		NumWorkers:     4,    // CPU threads
		Seed:           42,   // same seed = same result

		// 1. What does a candidate look like? Here: random text.
		Generator: func(rng *rand.Rand) []byte {
			g := make([]byte, len(target))
			for i := range g {
				g[i] = alphabet[rng.Intn(len(alphabet))]
			}
			return g
		},

		// 2. How good is a candidate? Here: number of correct letters.
		FitnessFunc: func(g []byte) float64 {
			score := 0.0
			for i := range g {
				if g[i] == target[i] {
					score++
				}
			}
			return score
		},

		// 3. How are parents chosen, combined and changed?
		Selector:  genetic.NewTournamentSelection[[]byte](3),
		Crossover: genetic.NewSinglePointCrossover[[]byte, byte](),
		Mutator:   genetic.NewByteMutation(alphabet),
	}

	engine, err := genetic.NewEngine(cfg)
	if err != nil {
		panic(err)
	}
	res, err := engine.Run() // 4. Run the evolution
	if err != nil {
		panic(err)
	}

	fmt.Printf("%s (score %.0f/%d, %d generations, %v)\n",
		res.BestIndividual.Genome, res.BestIndividual.Fitness, len(target),
		res.GenerationsEvaluated, res.TotalDuration)
}
```

A candidate can be any Go type (a list of yes/no choices, numbers, an ordering, a structure). Ready-made building blocks cover the common cases:

| Candidate is... | Crossover | Mutation | Examples |
|-----------------|-----------|----------|----------|
| Yes/no choices | single-, two-point, uniform | bit flip | knapsack |
| Real numbers | single-, two-point, uniform | Gaussian noise | function optimization, parameter fitting |
| An ordering (permutation) | order crossover (OX1) | swap, inversion | routes, sequencing |
| Text, or integer labels (e.g. task → machine) | single-, two-point, uniform | byte mutation (text), swap (labels) | the sentence above, scheduling |

Parent selection offers tournament, roulette-wheel and rank selection. Each of these is a small interface you can also implement yourself.

---

## What the library gives you

- **Uses all CPU cores**, for scoring and for creating offspring, with results **identical regardless of the number of threads**.
- **Reproducible science:** a fixed seed replays a run exactly; a random run reports its seed so it can be replayed later.
- **Island model:** several populations evolve separately and occasionally exchange their best candidates, which helps avoid getting stuck on a mediocre solution.
- **Control and monitoring:** progress statistics every generation (best, average, worst, spread), CSV export, and stop conditions (target score reached, no progress, time limit).
- **Safe by design:** a time limit or cancellation returns the best result so far; an error or crash inside your scoring function comes back as an ordinary error instead of killing the program.
- **No external dependencies**, only the Go standard library. About 97% of the code is covered by automated tests, which also run under Go's race detector and fuzzing.

See the **[usage guide](docs/usage.md)** for all options with code.

---

## How well does it work?

Two standard test functions are used by optimization researchers because they are full of misleading local minima: **Ackley** and **Rastrigin**, in 5 dimensions (the perfect answer scores 0). Every method got the same budget of 10,000 evaluations per run, and each was run 30 times with different seeds. A run counts as a *success* if it reaches a value below 0.1.

| Test function | Genetic algorithm | Simulated annealing | Random search |
|---------------|------------------:|--------------------:|--------------:|
| Ackley | **30/30** successes | 7/30 | 0/30 |
| Rastrigin | 2/30 (island model, 8 islands: **8/30**) | 0/30 | 0/30 |

On Rastrigin the genetic algorithm usually ends near the best answer without quite reaching it: its average final value is about 2.2 (1.1 with islands), against 16 for random search and 33 for simulated annealing.

**How to read this:** these are measurements on two functions with these settings, not a proof that genetic algorithms beat other methods in general. With only 30 runs, differences of a few successes are within noise (the same single-population configuration scored 5/30 with different random streams and 2/30 with others). Methodology, formulas, per-method tables and the parallel-speed measurements (up to about 5× faster on a 12-thread laptop CPU) are in [docs/experiments.md](docs/experiments.md).

---

## When not to use it

- If your function is smooth and you can compute its gradient, gradient-based methods are usually much faster.
- If the number of possible answers is small enough to try them all, do that: it guarantees the best one.
- A genetic algorithm gives a good answer, not a certificate that nothing better exists.

---

## Documentation

| Document | Contents |
|----------|----------|
| [docs/usage.md](docs/usage.md) | Operators, progress monitoring, reproducibility, cancellation, island model |
| [docs/architecture.md](docs/architecture.md) | Internal design, parallel execution model, design decisions |
| [docs/experiments.md](docs/experiments.md) | Benchmarks, comparisons with other methods, parallel scalability |
| [pkg.go.dev reference](https://pkg.go.dev/github.com/Semplicementeio/Genetic-Algorithm) | API reference |

**Development:** `go test -race ./...` runs the tests; `go test -run='^$' -fuzz=FuzzEngineConfig -fuzztime=10s ./genetic` runs the fuzzer; `go test -run='^$' -bench=. ./benchmarks` runs the benchmarks.

**Roadmap:** benchmark of the island topologies (ring / random / fully connected); adaptive mutation and diversity control; API stabilization for `v1.0.0`.

---

## Citation and license

If this software materially contributes to an academic or scientific work, please cite it with the metadata in [`CITATION.cff`](CITATION.cff):

```bibtex
@software{Genetic_Algorithm_2026,
  author  = {Semplicementeio},
  title   = {Genetic-Algorithm: A Genetic Algorithm Library for Go},
  year    = {2026},
  url     = {https://github.com/Semplicementeio/Genetic-Algorithm},
  version = {0.2.0}
}
```

Licensed under the **Genetic-Algorithm Custom Software License v1.0** ([`LICENSE`](LICENSE)). You may use, modify, distribute and commercially incorporate the software freely. Publicly released academic or scientific work that it materially contributes to must cite the project and its author (`Semplicementeio`).
