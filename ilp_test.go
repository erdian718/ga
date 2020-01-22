package ga_test

import (
	"math"
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
		return float64(f) - 100*math.Exp(float64(s)/10)
	} else {
		return float64(f)
	}
}

func (x ILP) Mutate() ga.Entity {
	return genILP()
}

func (x ILP) Crossover(e ga.Entity) ga.Entity {
	var z ILP
	y := e.(ILP)
	z[0] = math.Round(0.5*x[0] + 0.5*y[0])
	z[1] = math.Round(0.5*x[1] + 0.5*y[1])
	return z
}

func TestILP(t *testing.T) {
	m := ga.New(500, genILP)
	e, ok := m.Evolve(32, 100000)
	if !ok {
		t.FailNow()
	}

	f := m.Felite()

	println(f)
	x := e.(ILP)
	println(x[0], x[1])

	if math.Abs(f-e.Fitness()) > 1e-10 {
		t.FailNow()
	}
	if f < 40-5 {
		t.FailNow()
	}
}

func genILP() ga.Entity {
	mutex.Lock()
	defer mutex.Unlock()
	return ILP{float64(rnd.Intn(100)), float64(rnd.Intn(100))}
}
