# Experimental Evaluation of Evolutionary Optimization Strategies in Go

**Author**: Semplicementeio  
**Environment**: Go 1.22.2 / Linux x86_64 / AMD Ryzen 5 5500U (6 Cores, 12 Threads)  

---

## Abstract

This paper presents an empirical performance evaluation of the `github.com/Semplicementeio/Genetic-Algorithm` library. We analyze the scalability of parallel worker-pool fitness evaluations, measure generational throughput across population sizes, evaluate scientific reproducibility under deterministic PRNG seeds, and compare Genetic Algorithm (GA) convergence performance against Simulated Annealing (SA) and Random Search (RS) on non-convex multi-modal landscapes (Rastrigin & Ackley functions). In addition, we evaluate the impact of coarse-grained parallel Island Model optimization across varying island configurations.

---

## 1. Concurrency & Worker Scalability

`NumWorkers` parallelizes fitness evaluation and offspring creation. We benchmarked a 50-dimensional Ackley problem (1,000 individuals x 20 generations = 19,962 fitness evaluations) at two fitness costs, against a **hardware ceiling**: the same fitness work executed on bare goroutines with no GA code, i.e. the best any implementation could do on this machine (AMD Ryzen 5 5500U: 6 cores / 12 threads, clock drops under all-core load).

*Methodology: engine and ceiling measured back to back (interleaved) in one process, best of 7 runs per cell. The test machine is a laptop with some background load, so interleaving and best-of-N reduce, but do not eliminate, noise; expect absolute values to vary between runs.*

| Workers | Ceiling | Engine, moderate fitness (~28 µs) | Engine / ceiling | Engine, cheap fitness (~0.5 µs) |
|---------|---------|-----------------------------------|------------------|---------------------------------|
| 1  | 1.00x (553 ms) | 1.00x (578 ms) | – | 1.00x (39.7 ms) |
| 2  | 1.96x | 1.71x | 87% | 1.67x |
| 4  | 3.64x | 3.08x | 85% | 2.40x |
| 6  | 4.11x | 3.88x | 94% | 2.71x |
| 8  | 5.16x | 4.69x | 91% | 2.95x |
| 12 | 5.93x | 5.10x | 86% | 3.00x |

### Analysis

- **Moderate fitness**: the engine reaches 86-94% of the achievable speedup, with ~4% single-worker overhead over bare evaluation.
- **Cheap fitness**: fitness is only ~25% of the single-worker run; the rest is breeding, genome allocation and GC. Those phases are parallel but allocation-heavy, so speedup saturates near 3x.
- **Beyond 6 workers** gains come from SMT (12 hardware threads on 6 cores) and are inherently sub-linear; the ceiling column shows the same flattening.

### Root causes found in the previous implementation

A CPU profile of the original engine with cheap fitness showed:

1. **Channel contention**: 30-40% of CPU in `chanrecv`/`lock`/`procyield`, because each individual was handed to workers through one shared channel. Fixed with block scheduling over an atomic counter.
2. **Serial breeding**: selection, crossover and mutation ran on one goroutine. Fixed with chunked parallel breeding; each chunk owns a random stream derived from the seed, so determinism holds for any `NumWorkers`.
3. **Full-population sort per generation** to extract 2-3 elites (~8.5% of CPU). Replaced by an O(n*k) pass; rank and roulette selection now precompute once per generation.

The earlier table in this report (1 worker 5.56 s, 4 workers slower than 2) used a ~1 ms fitness - where serial engine code is <1% of the run, so Amdahl's Law could not explain it - and a single un-repeated measurement on a loaded machine. Those results were most likely measurement noise rather than a scaling law.

Before/after on one workload (pop 2,000, 12 workers): cheap fitness 69 ms -> 31 ms; ~40 µs fitness 273 ms -> 228 ms (speedup 3.9x -> 4.5x).

---

## 2. Population Sizing & Memory Throughput

We evaluated runtime and allocation overhead across varying population sizes (50 generations per run):

| Population Size | Time / Run (ms) | Allocations / Op | Memory / Op |
|-----------------|-----------------|------------------|-------------|
| 100             | 5.44 ms         | 10,220           | 1.01 MB     |
| 500             | 15.64 ms        | 46,733           | 4.89 MB     |
| 1000            | 26.10 ms        | 92,170           | 9.74 MB     |
| 5000            | 107.12 ms       | 454,527          | 48.42 MB    |

Observations show strict $O(N)$ linear memory scaling with respect to population size $N$.

---

## 3. Algorithm Comparison: Multi-Modal Function Optimization

We benchmarked GA against Simulated Annealing (SA) and Random Search (RS) across $N=30$ independent trials (Dimension $d=5$, Search Space $[-5.12, 5.12]^5$, $10,000$ evaluations per run, target threshold $< 0.10$, seeds 42–71).

### Rastrigin Function (Highly Multi-Modal)

$$f(\mathbf{x}) = 10d + \sum_{i=1}^d [x_i^2 - 10 \cos(2\pi x_i)]$$

| Algorithm | Mean ± StdDev | Best Min Found | Success Rate (<0.1) | Avg Time (ms) |
|-----------|---------------|----------------|---------------------|---------------|
| **GA**    | **2.1565 ± 1.1514** | **0.000240** | **2/30 (7%)**       | 4.63 ms       |
| **SA**    | 32.6073 ± 12.1240   | 3.979875     | 0/30 (0%)           | 1.54 ms       |
| **RS**    | 16.0248 ± 2.6549    | 10.113479    | 0/30 (0%)           | 1.75 ms       |

### Ackley Function (Global Basin with Multi-Local Optima)

$$f(\mathbf{x}) = -20 \exp\left(-0.2 \sqrt{\frac{1}{d} \sum x_i^2}\right) - \exp\left(\frac{1}{d} \sum \cos(2\pi x_i)\right) + 20 + e$$

| Algorithm | Mean ± StdDev | Best Min Found | Success Rate (<0.1) | Avg Time (ms) |
|-----------|---------------|----------------|---------------------|---------------|
| **GA**    | **0.0031 ± 0.0017** | **0.000440** | **30/30 (100%)**    | 5.26 ms       |
| **SA**    | 5.0892 ± 3.2166     | 0.000328     | 7/30 (23%)          | 2.06 ms       |
| **RS**    | 3.3152 ± 0.4087     | 2.441415     | 0/30 (0%)           | 2.06 ms       |

---

## 4. Island Model Evaluation (Single Pop vs 4 Islands vs 8 Islands)

Evaluating the effect of multi-island sub-populations on the Rastrigin 5D multi-modal landscape ($N=30$ independent trials, fixed 10,000 total evaluations):

| Architecture | Mean ± StdDev | Best Min Found | Success Rate (<0.1) | Avg Time (ms) |
|--------------|---------------|----------------|---------------------|---------------|
| Single Population GA (100 pop) | 2.1565 ± 1.1514 | 0.000240 | 2/30 (7%) | 5.04 ms |
| Island GA (4 Islands, 25 pop/island) | 1.1951 ± 1.0077 | 0.000511 | 8/30 (27%) | 3.02 ms |
| **Island GA (8 Islands, 12 pop/island)** | **1.0678 ± 0.8879** | **0.001053** | **27%** (8/30) | **3.66 ms** |

### Key Takeaways

1. **Diversity Preservation**: Dividing the global population into independent sub-populations (islands) can reduce premature convergence. Success rate on Rastrigin 5D is 27% with 4 or 8 islands versus 7% for a single population (8/30 vs 2/30 trials). With N=30 these counts are noisy: the same single-population GA scored 5/30 (17%) before the random streams of breeding were reorganized, so treat the difference as indicative, not conclusive.
2. **Exploration Trade-offs**: Gradient-based methods (Gradient Descent, L-BFGS) remain generally preferable when the objective is differentiable and problem structure allows efficient gradient computation. For sufficiently small discrete search spaces, exhaustive search may be simpler and faster while guaranteeing the optimum.
