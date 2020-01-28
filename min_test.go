package ga_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/ofunc/ga"
)

type MIN struct {
	X, Y float64
}

func (r MIN) Fitness() float64 {
	return -sqr(r.X) - sqr(r.Y)
}

func (r MIN) Mutate() ga.Entity {
	return genMIN()
}

func (r MIN) Crossover(e ga.Entity, w float64) ga.Entity {
	a := e.(MIN)
	return MIN{w*r.X + (1-w)*a.X, w*r.Y + (1-w)*a.Y}
}

func TestMIN(t *testing.T) {
	m := ga.New(100, genMIN)
	e, f, ok := m.Evolve(32, 100000)
	if e != m.Elite() {
		t.FailNow()
	}
	if math.Abs(f-e.Fitness()) > 1e-10 {
		t.FailNow()
	}
	if math.Abs(f-m.Fitness()) > 1e-10 {
		t.FailNow()
	}
	if !ok {
		t.FailNow()
	}

	if f < -1e-2 {
		t.Fatal("fitness(0):", f)
	}
	r := e.(MIN)
	if math.Abs(r.X) > 0.05 {
		t.Fatal("x(0):", r.X)
	}
	if math.Abs(r.Y) > 0.05 {
		t.Fatal("y(0):", r.Y)
	}
}

func genMIN() ga.Entity {
	return MIN{10*rand.Float64() - 5, 10*rand.Float64() - 5}
}
