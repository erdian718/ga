package ga_test

import (
	"math"
	"math/rand"
	"sync"
	"testing"
	"time"

	"github.com/ofunc/ga"
)

var (
	rnd   = rand.New(rand.NewSource(time.Now().Unix()))
	mutex = sync.Mutex{}
)

type Rosenbrock struct {
	X, Y float64
}

func (r Rosenbrock) Fitness() float64 {
	return -sqr(1-r.X) - 100*sqr(r.Y-sqr(r.X))
}

func (r Rosenbrock) Mutate() ga.Entity {
	return gen()
}

func (r Rosenbrock) Crossover(e ga.Entity, w float64) ga.Entity {
	a := e.(Rosenbrock)
	return Rosenbrock{w*r.X + (1-w)*a.X, w*r.Y + (1-w)*a.Y}
}

func TestRosenbrock(t *testing.T) {
	m := ga.New(1000, gen)
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

func BenchmarkRosenbrock(b *testing.B) {
	m := ga.New(1000, gen)
	for i := 0; i < b.N; i++ {
		m.Next()
	}
}

func sqr(x float64) float64 {
	return x * x
}

func gen() ga.Entity {
	mutex.Lock()
	defer mutex.Unlock()
	return Rosenbrock{10*rnd.Float64() - 5, 10*rnd.Float64() - 5}
}
