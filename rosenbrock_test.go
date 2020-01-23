package ga_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/ofunc/ga"
)

type Rosenbrock struct {
	X, Y float64
}

func (r Rosenbrock) Fitness() float64 {
	return -sqr(1-r.X) - 100*sqr(r.Y-sqr(r.X))
}

func (r Rosenbrock) Mutate() ga.Entity {
	return genRosenbrock()
}

func (r Rosenbrock) Crossover(e ga.Entity, w float64) ga.Entity {
	a := e.(Rosenbrock)
	return Rosenbrock{w*r.X + (1-w)*a.X, w*r.Y + (1-w)*a.Y}
}

func TestRosenbrock(t *testing.T) {
	m := ga.New(1000, genRosenbrock)
	e, f, ok := m.Evolve(32, 100000)
	if e != m.Elite() {
		t.FailNow()
	}
	if math.Abs(f-e.Fitness()) > 1e-10 {
		t.FailNow()
	}
	if !ok {
		t.FailNow()
	}

	if f < -1e-2 {
		t.FailNow()
	}
	r := e.(Rosenbrock)
	if math.Abs(r.X-1) > 0.1 {
		t.FailNow()
	}
	if math.Abs(r.Y-1) > 0.1 {
		t.FailNow()
	}
}

func genRosenbrock() ga.Entity {
	return Rosenbrock{10*rand.Float64() - 5, 10*rand.Float64() - 5}
}
