package ga_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/ofunc/ga"
)

type ILP [2]float64

func (x ILP) Fitness() float64 {
	s, f := 0.0, 5*x[0]+8*x[1]
	if d := x[0] + x[1] - 6; d > 0 {
		s += d
	}
	if d := 5*x[0] + 9*x[1] - 45; d > 0 {
		s += d
	}
	if s > 0 {
		return -s
	} else {
		return f
	}
}

func (x ILP) Mutate() ga.Entity {
	return genILP()
}

func (x ILP) Crossover(e ga.Entity, w float64) ga.Entity {
	y := e.(ILP)
	x[0] = math.Round(w*x[0] + (1-w)*y[0])
	x[1] = math.Round(w*x[1] + (1-w)*y[1])
	return x
}

func TestILP(t *testing.T) {
	m := ga.New(500, genILP)
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

	if f < 40-5 {
		t.Fatal("fitness(40):", f)
	}
}

func genILP() ga.Entity {
	return ILP{float64(rand.Intn(10)), float64(rand.Intn(10))}
}
