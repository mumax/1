package sim

import (
	"testing"
)

var Ns []int = []int{2, 8, 16, 31, 33, 128, 255, 511, 1023, 1024, 1025, 20000}

func TestSum(t *testing.T) {

	for _, N := range Ns {

		host := make([]float, N)
		for i := range host {
			host[i] = 1.
		}

		dev := backend.newArray(N)
		backend.memcpyTo(&(host[0]), dev, N)

		sum := NewSum(backend, N)
		result := sum.Reduce(dev)

		if result != float(N) {
			t.Error("expected ", N, " got ", result)
		}
	}
}

func TestMax(t *testing.T) {

	for _, N := range Ns {

		host := make([]float, N)
		for i := range host {
			host[i] = 1.
		}

		host[12349%N] = 10. //insert 10. in some quasi-random position

		dev := backend.newArray(N)
		backend.memcpyTo(&(host[0]), dev, N)

		max := NewMax(backend, N)
		result := max.Reduce(dev)

		if result != 10. {
			t.Error("expected ", 10., " got ", result)
		}
	}
}

func TestMaxAbs(t *testing.T) {

  for _, N := range Ns {

    host := make([]float, N)
    for i := range host {
      host[i] = 1.
    }

    host[12349%N] = -10. //insert 10. in some quasi-random position

    dev := backend.newArray(N)
    backend.memcpyTo(&(host[0]), dev, N)

    max := NewMaxAbs(backend, N)
    result := max.Reduce(dev)

    if result != 10. {
      t.Error("expected ", 10., " got ", result)
    }
  }
}

