// Package ga implements the genetic algorithm.
// It can handle negative fitness properly.
// Mutation probability is adaptive and does not need to be set.
package ga

import (
	"math"
	"math/rand"
	"sync"
	"time"
)

// Entity is an entity of GA model.
type Entity interface {
	// Fitness is the fitness of this entity.
	Fitness() float64
	// Mutate is the mutation operation.
	Mutate() Entity
	// Crossover is the crossover operation.
	Crossover(Entity, float64) Entity
}

// GA is a GA model.
type GA struct {
	n         int
	fitness   float64
	elite     Entity
	pm        float64
	std       float64
	base      float64
	fsum      float64
	rnd       *rand.Rand
	mutex     sync.Mutex
	fentities []float64
	entities  []Entity
	tentities []Entity
}

// New creates a GA model.
func New(n int, g func() Entity) *GA {
	m := &GA{
		n:         n,
		fitness:   math.Inf(-1),
		pm:        0.1,
		rnd:       rand.New(rand.NewSource(time.Now().Unix())),
		fentities: make([]float64, n),
		entities:  make([]Entity, n),
		tentities: make([]Entity, n),
	}
	for i := range m.entities {
		m.entities[i] = g()
	}
	m.adjust()
	m.base = m.std
	return m
}

// Fitness returns the fitness of current elite.
func (m *GA) Fitness() float64 {
	return m.fitness
}

// Elite returns the current elite.
func (m *GA) Elite() Entity {
	return m.elite
}

// RandInt returns, as an int, a random number in [0, n).
// It is safe for concurrent use by multiple goroutines.
func (m *GA) RandInt(n int) int {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	return m.rnd.Intn(n)
}

// RandFloat returns, as a float64, a random number in [0.0, 1.0).
// It is safe for concurrent use by multiple goroutines.
func (m *GA) RandFloat() float64 {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	return m.rnd.Float64()
}

// RandNorm returns a normally distributed float64 in the range [-math.MaxFloat64, +math.MaxFloat64]
// with standard normal distribution (mean = 0, stddev = 1).
// It is safe for concurrent use by multiple goroutines.
func (m *GA) RandNorm() float64 {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	return m.rnd.NormFloat64()
}

// RandExp returns an exponentially distributed float64 in the range (0, +math.MaxFloat64]
// with an exponential distribution whose rate parameter (lambda) is 1 and whose mean is 1/lambda (1).
// It is safe for concurrent use by multiple goroutines.
func (m *GA) RandExp() float64 {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	return m.rnd.ExpFloat64()
}

// Next gets the next generation of GA model, and returns the current elite and fitness.
func (m *GA) Next() (Entity, float64) {
	for i := range m.tentities {
		x, y, w := m.select2()
		z := x.Crossover(y, w)
		if m.rnd.Float64() < m.pm {
			z = z.Mutate()
		}
		m.tentities[i] = z
	}
	m.pm *= 0.2*math.Exp(-5*m.std/m.base) + 0.9
	m.entities, m.tentities = m.tentities, m.entities
	m.adjust()
	return m.elite, m.fitness
}

// Evolve runs the GA model until the elite k generations have not changed,
// or the max of iterations has been reached.
func (m *GA) Evolve(k int, max int) (Entity, float64, bool) {
	i, fitness := 0, m.fitness
	for j := 0; i < k && j < max; i, j = i+1, j+1 {
		_, f := m.Next()
		if fitness < f {
			i, fitness = 0, f
		}
	}
	return m.elite, fitness, i >= k
}

func (m *GA) adjust() {
	sm, sv := 0.0, 0.0
	for i, e := range m.entities {
		f := e.Fitness()
		sm, sv = sm+f, sv+f*f
		m.fentities[i] = f
		if m.fitness < f {
			m.fitness, m.elite = f, e
		}
	}
	mean, std := sm/float64(m.n), 1.0
	if v := sv/float64(m.n) - mean*mean; v > 0 {
		std = math.Sqrt(v)
	}

	m.fsum, m.std = 0, std
	for i, f := range m.fentities {
		f = 1 / (1 + math.Exp((mean-f)/std))
		m.fentities[i] = f
		m.fsum += f
	}
}

func (m *GA) select2() (Entity, Entity, float64) {
	rx, ry := m.rnd.Float64(), m.rnd.Float64()
	if rx > ry {
		rx, ry = ry, rx
	}
	fz, d, isx := m.fsum*rx, m.fsum*(ry-rx), true
	x, y, wx, wy := m.entities[0], m.entities[m.n-1], 0.0, 0.0
	for i, f := range m.fentities {
		if fz <= f {
			if isx {
				x, wx, isx = m.entities[i], f, false
				fz = fz + d - f*ry
				continue
			} else {
				y, wy = m.entities[i], f
				break
			}
		}
		fz -= f
	}
	return x, y, wx / (wx + wy)
}
