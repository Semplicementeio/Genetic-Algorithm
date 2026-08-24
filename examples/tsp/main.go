package main

import (
	"fmt"
	"math"
	"math/rand"
	"strings"

	"github.com/Semplicementeio/Genetic-Algorithm/genetic"
)

type City struct {
	ID   int
	Name string
	X, Y float64
}

func distance(c1, c2 City) float64 {
	dx := c1.X - c2.X
	dy := c1.Y - c2.Y
	return math.Sqrt(dx*dx + dy*dy)
}

func totalRouteDistance(route []int, cities []City) float64 {
	dist := 0.0
	for i := 0; i < len(route); i++ {
		from := cities[route[i]]
		to := cities[route[(i+1)%len(route)]]
		dist += distance(from, to)
	}
	return dist
}

func main() {
	cities := []City{
		{0, "Rome", 41.9, 12.5},
		{1, "Milan", 45.4, 9.1},
		{2, "Naples", 40.8, 14.2},
		{3, "Turin", 45.0, 7.6},
		{4, "Palermo", 38.1, 13.3},
		{5, "Genoa", 44.4, 8.9},
		{6, "Bologna", 44.5, 11.3},
		{7, "Florence", 43.7, 11.2},
		{8, "Bari", 41.1, 16.8},
		{9, "Catania", 37.5, 15.0},
		{10, "Venice", 45.4, 12.3},
		{11, "Verona", 45.4, 10.9},
		{12, "Messina", 38.1, 15.5},
		{13, "Padua", 45.4, 11.8},
		{14, "Trieste", 45.6, 13.7},
		{15, "Brescia", 45.5, 10.2},
		{16, "Taranto", 40.4, 17.2},
		{17, "Prato", 43.8, 11.0},
		{18, "Parma", 44.8, 10.3},
		{19, "Modena", 44.6, 10.9},
	}

	numCities := len(cities)

	fmt.Println("==================================================")
	fmt.Println("  🧬 Traveling Salesman Problem (TSP) Evolutionary Solver")
	fmt.Println("==================================================")
	fmt.Printf("Optimizing route across %d cities with Order Crossover (OX1)\n\n", numCities)

	cfg := genetic.Config[[]int]{
		PopulationSize: 200,
		Generations:    300,
		MutationRate:   0.15,
		CrossoverRate:  0.85,
		ElitismCount:   4,
		NumWorkers:     4,
		Seed:           42,
		Generator: func(rng *rand.Rand) []int {
			route := make([]int, numCities)
			for i := range route {
				route[i] = i
			}
			// Fisher-Yates shuffle
			rng.Shuffle(len(route), func(i, j int) {
				route[i], route[j] = route[j], route[i]
			})
			return route
		},
		FitnessFunc: func(route []int) float64 {
			d := totalRouteDistance(route, cities)
			if d == 0 {
				return 0.0
			}
			return 10000.0 / d
		},
		Selector:  genetic.NewTournamentSelection[[]int](4),
		Crossover: genetic.NewOrderCrossover[[]int, int](),
		Mutator:   genetic.NewInversionMutation[[]int, int](),
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

	bestRoute := res.BestIndividual.Genome
	finalDist := totalRouteDistance(bestRoute, cities)

	fmt.Println("Optimal Route Order:")
	cityNames := make([]string, len(bestRoute))
	for i, cityID := range bestRoute {
		cityNames[i] = cities[cityID].Name
	}
	fmt.Printf("  %s ➔ %s\n\n", strings.Join(cityNames, " ➔ "), cities[bestRoute[0]].Name)

	renderASCIIMap(bestRoute, cities)

	fmt.Println("--------------------------------------------------")
	fmt.Printf("Total Route Distance: %.2f km\n", finalDist)
	fmt.Printf("Generations Evaluated: %d\n", res.GenerationsEvaluated)
	fmt.Printf("Solver Execution Time: %v\n", res.TotalDuration)
	fmt.Println("==================================================")
}

func renderASCIIMap(route []int, cities []City) {
	width, height := 50, 15
	grid := make([][]rune, height)
	for i := range grid {
		grid[i] = make([]rune, width)
		for j := range grid[i] {
			grid[i][j] = ' '
		}
	}

	minX, maxX := cities[0].X, cities[0].X
	minY, maxY := cities[0].Y, cities[0].Y
	for _, c := range cities {
		if c.X < minX {
			minX = c.X
		}
		if c.X > maxX {
			maxX = c.X
		}
		if c.Y < minY {
			minY = c.Y
		}
		if c.Y > maxY {
			maxY = c.Y
		}
	}

	for _, c := range cities {
		gx := int((c.Y - minY) / (maxY - minY + 1e-5) * float64(width-1))
		gy := int((1.0 - (c.X-minX)/(maxX-minX+1e-5)) * float64(height-1))
		if gx >= 0 && gx < width && gy >= 0 && gy < height {
			grid[gy][gx] = '*'
		}
	}

	fmt.Println("ASCII Map Visualization (* = City Location):")
	fmt.Println("  +" + strings.Repeat("-", width) + "+")
	for i := range grid {
		fmt.Printf("  |%s|\n", string(grid[i]))
	}
	fmt.Println("  +" + strings.Repeat("-", width) + "+")
}
