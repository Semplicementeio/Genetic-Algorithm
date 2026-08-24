package main

import (
	"fmt"
	"math/rand"

	"github.com/Semplicementeio/Genetic-Algorithm/genetic"
)

type Task struct {
	ID       int
	Name     string
	Duration int
}

func main() {
	tasks := []Task{
		{0, "Database Migration", 45},
		{1, "Index Rebuild", 30},
		{2, "Cache Warming", 15},
		{3, "Report Generation", 60},
		{4, "ETL Ingestion", 90},
		{5, "Log Archival", 20},
		{6, "Model Retraining", 120},
		{7, "Security Audit", 40},
		{8, "Backup Creation", 50},
		{9, "Metrics Aggregation", 25},
	}

	numWorkers := 3 // 3 parallel machines

	fmt.Println("==================================================")
	fmt.Println("  🧬 Job Shop / Task Scheduling Optimization")
	fmt.Println("==================================================")
	fmt.Printf("Distributing %d tasks across %d worker machines to minimize makespan\n\n", len(tasks), numWorkers)

	cfg := genetic.Config[[]int]{
		PopulationSize: 100,
		Generations:    150,
		MutationRate:   0.10,
		CrossoverRate:  0.80,
		ElitismCount:   2,
		NumWorkers:     4,
		Seed:           42,
		Generator: func(rng *rand.Rand) []int {
			schedule := make([]int, len(tasks))
			for i := range schedule {
				schedule[i] = rng.Intn(numWorkers)
			}
			return schedule
		},
		FitnessFunc: func(schedule []int) float64 {
			machineLoads := make([]int, numWorkers)
			for taskIdx, machineID := range schedule {
				machineLoads[machineID] += tasks[taskIdx].Duration
			}

			// Find makespan (maximum completion time among machines)
			maxLoad := 0
			for _, load := range machineLoads {
				if load > maxLoad {
					maxLoad = load
				}
			}

			if maxLoad == 0 {
				return 0.0
			}
			// Minimize makespan
			return 10000.0 / float64(maxLoad)
		},
		Selector:  genetic.NewTournamentSelection[[]int](3),
		Crossover: genetic.NewSinglePointCrossover[[]int, int](),
		Mutator:   genetic.NewSwapMutation[[]int, int](),
		GenomeCopier: func(g []int) []int {
			c := make([]int, len(g))
			copy(c, g)
			return c
		},
	}

	engine, err := genetic.NewEngine(cfg)
	if err != nil {
		panic(err)
	}

	res, err := engine.Run()
	if err != nil {
		panic(err)
	}

	bestSchedule := res.BestIndividual.Genome
	machineLoads := make([]int, numWorkers)
	machineTasks := make([][]Task, numWorkers)

	for taskIdx, machineID := range bestSchedule {
		t := tasks[taskIdx]
		machineLoads[machineID] += t.Duration
		machineTasks[machineID] = append(machineTasks[machineID], t)
	}

	fmt.Println("Optimal Machine Task Assignment:")
	maxLoad := 0
	for m := 0; m < numWorkers; m++ {
		fmt.Printf(" Machine %d (Total Load: %d mins):\n", m+1, machineLoads[m])
		for _, t := range machineTasks[m] {
			fmt.Printf("   - %-22s (%d mins)\n", t.Name, t.Duration)
		}
		if machineLoads[m] > maxLoad {
			maxLoad = machineLoads[m]
		}
	}

	fmt.Println("--------------------------------------------------")
	fmt.Printf("Optimal Makespan (Completion Time): %d minutes\n", maxLoad)
	fmt.Printf("Generations Evaluated            : %d\n", res.GenerationsEvaluated)
	fmt.Printf("Execution Time                   : %v\n", res.TotalDuration)
	fmt.Println("==================================================")
}
