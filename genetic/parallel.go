package genetic

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"runtime/debug"
	"sync"
	"sync/atomic"
)

var (
	// ErrPanicRecovered wraps a panic raised inside a user-supplied function
	// (fitness, selector, crossover, mutator, copier) while the engine was running.
	ErrPanicRecovered = errors.New("panic recovered in user-supplied function")
	// ErrNaNFitness is returned when a fitness function yields NaN, which cannot be ordered.
	ErrNaNFitness = errors.New("fitness function returned NaN")
)

// splitMix64 is a tiny, fast rand.Source64 with O(1) seeding. It drives the
// per-chunk random streams used during parallel breeding, where creating a
// standard rand.NewSource (607-word state) for every chunk would be too costly.
type splitMix64 struct{ state uint64 }

func (s *splitMix64) Seed(seed int64) { s.state = uint64(seed) }

func (s *splitMix64) Uint64() uint64 {
	s.state += 0x9e3779b97f4a7c15
	z := s.state
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

func (s *splitMix64) Int63() int64 { return int64(s.Uint64() >> 1) }

func newChunkRand(seed int64) *rand.Rand {
	return rand.New(&splitMix64{state: uint64(seed)})
}

// runParallel calls fn(i) for every i in [0, n) using up to `workers` goroutines.
//
// Indices are claimed in blocks of `grain` through a single atomic counter, so
// there is no per-item channel traffic or lock contention. Each index is passed
// to fn exactly once, which lets fn write to distinct slice elements without
// further synchronization. The first error (including a recovered panic) or a
// canceled context stops the remaining work; that error is returned.
func runParallel(ctx context.Context, n, workers, grain int, fn func(i int) error) error {
	if n <= 0 {
		return nil
	}
	if grain < 1 {
		grain = 1
	}
	if workers > (n+grain-1)/grain {
		workers = (n + grain - 1) / grain
	}

	var (
		next     atomic.Int64
		stop     atomic.Bool
		errOnce  sync.Once
		firstErr error
	)
	fail := func(err error) {
		errOnce.Do(func() { firstErr = err })
		stop.Store(true)
	}

	work := func() {
		defer func() {
			if r := recover(); r != nil {
				fail(fmt.Errorf("%w: %v\n%s", ErrPanicRecovered, r, debug.Stack()))
			}
		}()
		for !stop.Load() {
			if err := ctx.Err(); err != nil {
				fail(err)
				return
			}
			start := int(next.Add(int64(grain))) - grain
			if start >= n {
				return
			}
			end := start + grain
			if end > n {
				end = n
			}
			for i := start; i < end; i++ {
				if err := fn(i); err != nil {
					fail(err)
					return
				}
			}
		}
	}

	if workers <= 1 {
		work()
		return firstErr
	}

	var wg sync.WaitGroup
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			work()
		}()
	}
	wg.Wait()
	return firstErr
}
