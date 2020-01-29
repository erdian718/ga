package ga_test

import (
	"math/rand"
	"testing"
	"time"

	"github.com/ofunc/ga"
)

type Benchmark float64

func (b Benchmark) Fitness() float64 {
	return -sqr(float64(b))
}

func (b Benchmark) Mutate() ga.Entity {
	return Benchmark(200*rand.Float64() - 100)
}

func (b Benchmark) Crossover(e ga.Entity, w float64) ga.Entity {
	x := float64(e.(Benchmark))
	return Benchmark(w*float64(b) + (1-w)*x)
}

func init() {
	rand.Seed(time.Now().Unix())
}

func BenchmarkGA(b *testing.B) {
	m := ga.New(10000, Benchmark(0).Mutate)
	for i := 0; i < b.N; i++ {
		m.Next()
	}
}

func sqr(x float64) float64 {
	return x * x
}
