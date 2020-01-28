// Package ga implements the genetic algorithm.
// It can handle negative fitness properly.
// Mutation probability is adaptive and does not need to be set.
package ga

import (
	"math"
	"math/rand"
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

// Next gets the next generation of GA model, and returns the current elite and fitness.
func (m *GA) Next() (Entity, float64) {
	m.pm *= 0.2*math.Exp(-5*m.std/m.base) + 0.9
	for i := range m.tentities {
		x, y, w := m.select2()
		z := x.Crossover(y, w)
		if m.rnd.Float64() < m.pm {
			z = z.Mutate()
		}
		m.tentities[i] = z
	}
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
	mean, std2 := 0.0, 0.0
	for i, e := range m.entities {
		f := e.Fitness()
		m.fentities[i] = f

		k1 := 1 / float64(i+1)
		k2 := 1 - k1
		mean = k1*f + k2*mean
		std2 = k1*f*f + k2*std2

		if m.fitness < f {
			m.fitness, m.elite = f, e
		}
	}
	std := 1.0
	std2 -= mean * mean
	if std2 > 0 {
		std = math.Sqrt(std2)
	}

	m.fsum = 0
	for i, f := range m.fentities {
		f = 1 / (1 + math.Exp((mean-f)/std))
		m.fentities[i] = f
		m.fsum += f
	}
	m.std = std
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
