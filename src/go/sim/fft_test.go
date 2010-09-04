package sim

import (
	"testing"
	"tensor"
 	"os"
	"fmt"
)



func TestFFT(t *testing.T) {


	sizes := [][]int{
// 		[]int{1, 32, 64},
	  []int{2, 4, 8}}

	for _, size := range sizes {
    
		fft := NewFFT(backend, size)
		fmt.Println(fft)
		outsize := fft.PhysicSize()

		devIn, devOut := NewTensor(backend, size), NewTensor(backend, outsize)
		host1, host2 := tensor.NewTensorN(size), tensor.NewTensorN(size)

		for i := 0; i < tensor.N(host1); i++ {
			host1.List()[i] = float(i%100) / 100
		}

		host1.List()[0] = 1.

    
		TensorCopyTo(host1, devIn)
		tensor.Format(os.Stdout, devIn)
		fft.Forward(devIn, devOut)
		tensor.Format(os.Stdout, devOut)
		fft.Inverse(devOut, devIn)
		tensor.Format(os.Stdout, devIn)
		TensorCopyFrom(devIn, host2)

		N := float(fft.Normalization())
		var maxError float = 0
		for i := range host2.List() {
			host2.List()[i] /= N
			if abs(host2.List()[i]-host1.List()[i]) > maxError {
				maxError = abs(host2.List()[i] - host1.List()[i])
			}
		}
		//tensor.Format(os.Stdout, host2)
		fmt.Println("FFT error:", maxError)
		if maxError > 1E-4 {
			t.Fail()
		}
	}

}

// func abs(r float) float {
// 	if r < 0 {
// 		return -r
// 	}
// 	//else
// 	return r
// }
