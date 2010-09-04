package sim

import (
	"testing"
	"tensor"
	//     "fmt"
	//     "os"
)

func TestConv(t *testing.T) {

	size4D := []int{3, 32, 32, 2}
	size := size4D[1:]
	//kernelSize := []int{2*size[X], 2*size[Y], 2*size[Z]}

	kernel := FaceKernel6(size, []float{1., 1., 1.}, 8)
	conv := NewConv(backend, size, kernel)

	//   for i,k:= range kernel{
	//     fmt.Println("kernel", i, k.Size())
	//   }
	//   fmt.Println("conv", conv.KernelSize())
	//   fmt.Println("conv.fft", conv.fft)

	conv.LoadKernel6(kernel)

	//   for i:=range(conv.kernel){
	//     fmt.Println(i)
	//     if conv.kernel[i] == nil {
	//       fmt.Println("(nil)")
	//     }else{
	//       tensor.Format(os.Stdout, conv.kernel[i])
	//     }
	//   }

	m, h := NewTensor(backend, size4D), NewTensor(backend, size4D)

	m.Set([]int{0, 7, 7, 0}, 1.)
	tensor.WriteFile("m.t", m)
	conv.Convolve(m, h)
	tensor.WriteFile("h.t", h)
}
