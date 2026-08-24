package main

import (
	"fmt"
	"math/rand"

	"github.com/Semplicementeio/Genetic-Algorithm/genetic"
)

type Item struct {
	Name   string
	Weight float64
	Value  float64
}

func main() {
	items := []Item{
		{"Laptop", 2.5, 1200},
		{"Camera", 1.2, 700},
		{"Drone", 3.0, 950},
		{"Smartphone", 0.4, 800},
		{"Tablet", 0.8, 450},
		{"Smartwatch", 0.2, 300},
		{"Headphones", 0.5, 200},
		{"Powerbank", 0.9, 150},
		{"Book", 0.6, 50},
		{"Thermos", 0.7, 40},
		{"Jacket", 1.5, 180},
		{"FirstAidKit", 1.0, 120},
	}

	maxCapacity := 6.0 // max 6.0 kg

	fmt.Println("==================================================")
	fmt.Println("  🧬 0/1 Knapsack Evolutionary Optimization")
	fmt.Println("==================================================")
	fmt.Printf("Total Available Items: %d | Max Capacity: %.1f kg\n\n", len(items), maxCapacity)

	cfg := genetic.Config[[]bool]{
		PopulationSize: 100,
		Generations:    100,
		MutationRate:   0.05,
		CrossoverRate:  0.8,
		ElitismCount:   2,
		NumWorkers:     4,
		Seed:           42,
		Generator: func(rng *rand.Rand) []bool {
			genome := make([]bool, len(items))
			for i := range genome {
				genome[i] = rng.Float64() < 0.3 // ~30% chance to select item initially
			}
			return genome
		},
		FitnessFunc: func(genome []bool) float64 {
			var totalWeight, totalValue float64
			for i, selected := range genome {
				if selected {
					totalWeight += items[i].Weight
					totalValue += items[i].Value
				}
			}

			if totalWeight > maxCapacity {
				// Penalty for exceeding capacity
				return 0.0
			}
			return totalValue
		},
		Selector:  genetic.NewTournamentSelection[[]bool](3),
		Crossover: genetic.NewSinglePointCrossover[[]bool, bool](),
		Mutator:   genetic.NewBitFlipMutation(),
	}

	engine, err := genetic.NewEngine(cfg)
	if err != nil {
		panic(err)
	}

	res, err := engine.Run()
	if err != nil {
		panic(err)
	}

	var totalWeight, totalValue float64
	fmt.Println("Optimal Item Selection:")
	for i, selected := range res.BestIndividual.Genome {
		if selected {
			fmt.Printf("  [✓] %-15s (Weight: %.1f kg, Value: $%.0f)\n", items[i].Name, items[i].Weight, items[i].Value)
			totalWeight += items[i].Weight
			totalValue += items[i].Value
		}
	}

	fmt.Println("--------------------------------------------------")
	fmt.Printf("Total Weight : %.1f kg / %.1f kg\n", totalWeight, maxCapacity)
	fmt.Printf("Total Value  : $%.0f\n", totalValue)
	fmt.Printf("Generations  : %d\n", res.GenerationsEvaluated)
	fmt.Printf("Elapsed Time : %v\n", res.TotalDuration)
	fmt.Printf("Termination  : %s\n", res.ConvergedReason)
	fmt.Println("==================================================")
}
