# Experimental Evaluation of Evolutionary Optimization Strategies in Go

**Author**: Semplicementeio  
**Environment**: Go 1.22.2 / Linux x86_64 / AMD Ryzen 5 5500U (6 Cores, 12 Threads)  

---

## Abstract

This paper presents an empirical performance evaluation of the `github.com/Semplicementeio/Genetic-Algorithm` library. We analyze the scalability of parallel worker-pool fitness evaluations, measure generational throughput across population sizes, evaluate scientific reproducibility under deterministic PRNG seeds, and compare Genetic Algorithm (GA) convergence performance against Simulated Annealing (SA) and Random Search (RS) on non-convex multi-modal landscapes (Rastrigin & Ackley functions). In addition, we evaluate the impact of coarse-grained parallel Island Model optimization across varying island configurations.

---

## 1. Concurrency & Worker-Pool Scalability

To measure the speedup of multi-threaded fitness evaluation, we benchmarked a heavy 50-dimensional optimization task (`Ackley` function with synthetic workload) across varying worker counts.

### Empirical Results (1000 Individuals, 20 Generations)

| Workers | Execution Time (ns/op) | Throughput Speedup | Allocations / Op | Memory / Op |
|---------|------------------------|-------------------|------------------|-------------|
| 1       | 5,561,900,811 ns       | **1.00x** (Base)  | 35,390           | 16.28 MB    |
| 2       | 2,868,093,249 ns       | **1.94x**         | 35,512           | 17.26 MB    |
| 4       | 3,113,030,915 ns       | **1.79x**         | 35,555           | 17.26 MB    |
| 8       | 2,148,099,920 ns       | **2.58x**         | 35,641           | 17.27 MB    |
| 16      | 1,807,508,746 ns       | **3.08x**         | 35,839           | 17.30 MB    |

### Analysis & Amdahl's Law

- **Linear Speedup up to 2 Workers**: Going from 1 to 2 workers achieved **1.94x** speedup (97% parallel efficiency).
- **Diminishing Returns (>8 Workers)**: Beyond physical core count (6 physical cores), thread context switching, channel synchronization overhead, and CPU cache line bouncing prevent linear scaling. The non-parallel portion of the algorithm (selection, crossover, mutation, elitism) imposes an upper bound on speedup per **Amdahl's Law**.

---

## 2. Population Sizing & Memory Throughput

We evaluated runtime and allocation overhead across varying population sizes (50 generations per run):

| Population Size | Time / Run (ms) | Allocations / Op | Memory / Op |
|-----------------|-----------------|------------------|-------------|
| 100             | 4.42 ms         | 9,430            | 1.38 MB     |
| 500             | 21.38 ms        | 45,041           | 6.83 MB     |
| 1000            | 47.35 ms        | 89,702           | 13.66 MB    |
| 5000            | 243.68 ms       | 446,681          | 68.27 MB    |

Observations show strict $O(N)$ linear memory scaling with respect to population size $N$.

---

## 3. Algorithm Comparison: Multi-Modal Function Optimization

We benchmarked GA against Simulated Annealing (SA) and Random Search (RS) across $N=30$ independent trials (Dimension $d=5$, Search Space $[-5.12, 5.12]^5$, $10,000$ evaluations per run, target threshold $< 0.10$, seeds 42–71).

### Rastrigin Function (Highly Multi-Modal)

$$f(\mathbf{x}) = 10d + \sum_{i=1}^d [x_i^2 - 10 \cos(2\pi x_i)]$$

| Algorithm | Mean ± StdDev | Best Min Found | Success Rate (<0.1) | Avg Time (ms) |
|-----------|---------------|----------------|---------------------|---------------|
| **GA**    | **1.8579 ± 1.2776** | **0.000139** | **5/30 (17%)**      | 6.31 ms       |
| **SA**    | 32.6073 ± 12.1240   | 3.979875     | 0/30 (0%)           | 1.54 ms       |
| **RS**    | 16.0248 ± 2.6549    | 10.113479    | 0/30 (0%)           | 1.75 ms       |

### Ackley Function (Global Basin with Multi-Local Optima)

$$f(\mathbf{x}) = -20 \exp\left(-0.2 \sqrt{\frac{1}{d} \sum x_i^2}\right) - \exp\left(\frac{1}{d} \sum \cos(2\pi x_i)\right) + 20 + e$$

| Algorithm | Mean ± StdDev | Best Min Found | Success Rate (<0.1) | Avg Time (ms) |
|-----------|---------------|----------------|---------------------|---------------|
| **GA**    | **0.0028 ± 0.0011** | **0.001100** | **30/30 (100%)**    | 6.61 ms       |
| **SA**    | 5.0892 ± 3.2166     | 0.000328     | 7/30 (23%)          | 2.06 ms       |
| **RS**    | 3.3152 ± 0.4087     | 2.441415     | 0/30 (0%)           | 2.06 ms       |

---

## 4. Island Model Evaluation (Single Pop vs 4 Islands vs 8 Islands)

Evaluating the effect of multi-island sub-populations on the Rastrigin 5D multi-modal landscape ($N=30$ independent trials, fixed 10,000 total evaluations):

| Architecture | Mean ± StdDev | Best Min Found | Success Rate (<0.1) | Avg Time (ms) |
|--------------|---------------|----------------|---------------------|---------------|
| Single Population GA (100 pop) | 1.8579 ± 1.2776 | 0.000139 | 5/30 (17%) | 6.05 ms |
| Island GA (4 Islands, 25 pop/island) | 1.3608 ± 1.0738 | 0.000291 | 6/30 (20%) | 3.25 ms |
| **Island GA (8 Islands, 12 pop/island)** | **1.1932 ± 1.2433** | **0.000466** | **37%** (11/30) | **3.56 ms** |

### Key Takeaways

1. **Diversity Preservation**: Dividing the global population into independent sub-populations (islands) significantly reduces premature convergence. Success rate on Rastrigin 5D increases from 17% to 37%.
2. **Exploration Trade-offs**: Gradient-based methods (Gradient Descent, L-BFGS) remain generally preferable when the objective is differentiable and problem structure allows efficient gradient computation. For sufficiently small discrete search spaces, exhaustive search may be simpler and faster while guaranteeing the optimum.
