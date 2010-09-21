package sim

import (
	"testing"
)

func TestSum(t *testing.T) {
	Ns := []int{2, 8, 16, 31, 33, 128, 255, 511, 1023, 1024, 1025, 20000}
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
