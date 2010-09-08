package sim

import (
	"testing"
	"tensor"
	"fmt"
	"os"
)

func TestConv(t *testing.T) {

	size4D := []int{3, 1, 8, 8}
	size := size4D[1:]
	kernelSize := padSize(size)

	kernel := FaceKernel6(kernelSize, []float{1., 1., 1.}, 8)
	conv := NewConv(backend, size, kernel)

	for i := range conv.kernel {
		fmt.Println("K", i)
		if conv.kernel[i] != nil {
			tensor.Format(os.Stdout, conv.kernel[i])
		}
	}

	m, h := NewTensor(backend, size4D), NewTensor(backend, size4D)

	m.Set([]int{0, 0, 7, 7}, 1.)
	tensor.WriteFile("m.t", m)
	conv.Convolve(m, h)
	tensor.WriteFile("h.t", h)
}
