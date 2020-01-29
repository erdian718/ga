package ga_test

import (
	"fmt"
	"math/rand"

	"github.com/ofunc/ga"
)

// V is an entity of GA model.
type V struct {
	X, Y float64
}

// Fitness is the fitness of this entity.
// The problem to be solved is:
// 	max = 2x + 3y
// 	4x + 3y <= 10
// 	3x + 5y <= 12
// 	x, y >= 0
func (v V) Fitness() float64 {
	s, f := 0.0, 2*v.X+3*v.Y
	if d := (4*v.X + 3*v.Y) / 10; d > 1 {
		s += d - 1
	}
	if d := (3*v.X + 5*v.Y) / 12; d > 1 {
		s += d - 1
	}
	if s > 0 {
		return -s
	} else {
		return f
	}
}

// Mutate is the mutation operation.
func (v V) Mutate() ga.Entity {
	return V{10 * rand.Float64(), 10 * rand.Float64()}
}

// Crossover is the crossover operation.
func (v V) Crossover(e ga.Entity, w float64) ga.Entity {
	a := e.(V)
	return V{w*v.X + (1-w)*a.X, w*v.Y + (1-w)*a.Y}
}

func Example() {
	m := ga.New(1000, V{}.Mutate)
	e, f, ok := m.Evolve(30, 10000)
	fmt.Println("fitness: ", f)
	fmt.Println("elite:", e)
	fmt.Println("isok:", ok)
}
