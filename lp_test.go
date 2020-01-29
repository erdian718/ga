package ga_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/ofunc/ga"
)

type LP [2]float64

func (x LP) Fitness() float64 {
	s, f := 0.0, 72*x[0]+64*x[1]
	if d := (x[0] + x[1]) / 50; d > 1 {
		s += d - 1
	}
	if d := (12*x[0] + 8*x[1]) / 480; d > 1 {
		s += d - 1
	}
	if d := (3 * x[0]) / 100; d > 1 {
		s += d - 1
	}
	if s > 0 {
		return -s
	} else {
		return f
	}
}

func (x LP) Mutate() ga.Entity {
	return LP{100 * rand.Float64(), 100 * rand.Float64()}
}

func (x LP) Crossover(e ga.Entity, w float64) ga.Entity {
	y := e.(LP)
	x[0] = math.Round(w*x[0] + (1-w)*y[0])
	x[1] = math.Round(w*x[1] + (1-w)*y[1])
	return x
}

func TestLP(t *testing.T) {
	m := ga.New(100, LP{}.Mutate)
	e, f, ok := m.Evolve(30, 100)
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

	if f < 3360-100 {
		t.Fatal("fitness(3360):", f)
	}
}
