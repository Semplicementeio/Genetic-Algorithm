package main

import (
	"bytes"
	"flag"
	"fmt"
	"math/rand"
	"os"

	"github.com/Semplicementeio/Genetic-Algorithm/examples/knapsack"
	"github.com/Semplicementeio/Genetic-Algorithm/examples/optimization"
	"github.com/Semplicementeio/Genetic-Algorithm/examples/scheduling"
	"github.com/Semplicementeio/Genetic-Algorithm/examples/tsp"
	"github.com/Semplicementeio/Genetic-Algorithm/genetic"
)

func main() {
	mode := flag.String("example", "all", "Example to run: knapsack, tsp, optimization, scheduling, or all")
	seed := flag.Int64("seed", 42, "PRNG random seed for scientific reproducibility")
	exportCSV := flag.Bool("export-csv", false, "Export generational progress stats to CSV file")
	flag.Parse()

	fmt.Println("==========================================================================================")
	fmt.Println("  🧬 Production-Quality Extensible Genetic Algorithm Library for Go")
	fmt.Println("  github.com/Semplicementeio/Genetic-Algorithm")
	fmt.Println("==========================================================================================")
	fmt.Printf("Configuration: Seed=%d | Selected Mode: %s\n\n", *seed, *mode)

	switch *mode {
	case "knapsack":
		knapsack.Run(*seed)
	case "tsp":
		tsp.Run(*seed)
	case "optimization":
		optimization.Run(*seed)
	case "scheduling":
		scheduling.Run(*seed)
	case "all":
		runQuickShowcase(*seed, *exportCSV)
	default:
		fmt.Printf("Unknown example: %s. Options: knapsack, tsp, optimization, scheduling, all\n", *mode)
		os.Exit(1)
	}
}

func runQuickShowcase(seed int64, exportCSV bool) {
	fmt.Println("--- 1. Quick Target String Evolution Showcase ---")
	target := "PRODUCTION READY GO GA"
	alphabet := "ABCDEFGHIJKLMNOPQRSTUVWXYZ "

	cfg := genetic.Config[[]byte]{
		PopulationSize: 150,
		Generations:    200,
		MutationRate:   0.04,
		CrossoverRate:  0.85,
		ElitismCount:   3,
		NumWorkers:     4,
		Seed:           seed,
		Generator: func(rng *rand.Rand) []byte {
			res := make([]byte, len(target))
			for i := range res {
				res[i] = alphabet[rng.Intn(len(alphabet))]
			}
			return res
		},
		FitnessFunc: func(g []byte) float64 {
			fit := 0.0
			for i := range g {
				if g[i] == target[i] {
					fit += 1.0
				}
			}
			return fit
		},
		Selector:  genetic.NewTournamentSelection[[]byte](3),
		Crossover: genetic.NewSinglePointCrossover[[]byte, byte](),
		Mutator:   genetic.NewByteMutation(alphabet),
		GenomeCopier: func(g []byte) []byte {
			c := make([]byte, len(g))
			copy(c, g)
			return c
		},
	}

	engine, err := genetic.NewEngine(cfg)
	if err != nil {
		fmt.Printf("Engine initialization failed: %v\n", err)
		return
	}

	res, err := engine.Run()
	if err != nil {
		fmt.Printf("Engine execution failed: %v\n", err)
		return
	}

	fmt.Printf(" Target String      : \"%s\"\n", target)
	fmt.Printf(" Evolved Solution   : \"%s\"\n", string(res.BestIndividual.Genome))
	fmt.Printf(" Best Fitness Score : %.0f / %d\n", res.BestIndividual.Fitness, len(target))
	fmt.Printf(" Generations Run    : %d\n", res.GenerationsEvaluated)
	fmt.Printf(" Execution Time     : %v\n", res.TotalDuration)
	fmt.Printf(" Termination Reason : %s\n", res.ConvergedReason)
	fmt.Println()

	if exportCSV {
		var buf bytes.Buffer
		if err := res.History.ExportCSV(&buf); err == nil {
			_ = os.WriteFile("evolution_stats.csv", buf.Bytes(), 0644)
			fmt.Println(" Saved generational stats log to 'evolution_stats.csv'")
		}
	}

	fmt.Println("--- 2. Executing Real-World Domain Problem Examples ---")
	fmt.Println("Running 0/1 Knapsack problem...")
	knapsack.Run(seed)
	fmt.Println()

	fmt.Println("Running Traveling Salesman Problem (TSP)...")
	tsp.Run(seed)
	fmt.Println()

	fmt.Println("Running Mathematical Function Benchmark & Algorithm Comparison...")
	optimization.Run(seed)
	fmt.Println()

	fmt.Println("Running Job Shop Task Scheduling...")
	scheduling.Run(seed)
	fmt.Println()
}
