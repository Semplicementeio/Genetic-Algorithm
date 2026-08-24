package genetic

// Individual represents a single candidate solution in the population.
// T can be any data structure (e.g. []byte, []bool, []int, []float64, custom struct).
type Individual[T any] struct {
	Genome    T
	Fitness   float64
	Evaluated bool
}

// Clone creates a shallow or deep copy depending on whether a copier is provided.
func (ind Individual[T]) Clone(copier func(T) T) Individual[T] {
	newGenome := ind.Genome
	if copier != nil {
		newGenome = copier(ind.Genome)
	}
	return Individual[T]{
		Genome:    newGenome,
		Fitness:   ind.Fitness,
		Evaluated: ind.Evaluated,
	}
}

// Population represents a collection of individuals.
type Population[T any] []Individual[T]

// Best returns the individual with the highest fitness in the population.
func (p Population[T]) Best() Individual[T] {
	if len(p) == 0 {
		return Individual[T]{}
	}
	best := p[0]
	for i := 1; i < len(p); i++ {
		if p[i].Fitness > best.Fitness {
			best = p[i]
		}
	}
	return best
}

// AverageFitness computes the mean fitness of the population.
func (p Population[T]) AverageFitness() float64 {
	if len(p) == 0 {
		return 0
	}
	total := 0.0
	for _, ind := range p {
		total += ind.Fitness
	}
	return total / float64(len(p))
}

// Worst returns the individual with the lowest fitness in the population.
func (p Population[T]) Worst() Individual[T] {
	if len(p) == 0 {
		return Individual[T]{}
	}
	worst := p[0]
	for i := 1; i < len(p); i++ {
		if p[i].Fitness < worst.Fitness {
			worst = p[i]
		}
	}
	return worst
}
