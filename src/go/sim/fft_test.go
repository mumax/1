//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import (
	"testing"
	"tensor"
	"fmt"
	_ "rand"
)

var fft_test_sizes [][]int = [][]int{
	[]int{1, 4, 8}} //,
// 	[]int{2, 4, 8}}


func TestFFTPadded(t *testing.T) {

	for _, size := range fft_test_sizes {

		paddedsize := padSize(size)

		fft := NewFFTPadded(backend, size, paddedsize)
		fftP := NewFFT(backend, paddedsize) // with manual padding

		fmt.Println(fft)
		fmt.Println(fftP)

		outsize := fftP.PhysicSize()

		dev, devT, devTT := NewTensor(backend, size), NewTensor(backend, outsize), NewTensor(backend, size)
		devP, devPT, devPTT := NewTensor(backend, paddedsize), NewTensor(backend, outsize), NewTensor(backend, paddedsize)

		host, hostT, hostTT := tensor.NewTensorN(size), tensor.NewTensorN(outsize), tensor.NewTensorN(size)
		hostP, hostPT, hostPTT := tensor.NewTensorN(paddedsize), tensor.NewTensorN(outsize), tensor.NewTensorN(paddedsize)

		for i := 0; i < size[0]; i++ {
			for j := 0; j < size[1]; j++ {
				for k := 0; k < size[2]; k++ {
					host.List()[i*size[1]*size[2]+j*size[2]+k] = 1. //rand.Float() //1.
					hostP.List()[i*paddedsize[1]*paddedsize[2]+j*paddedsize[2]+k] = host.List()[i*size[1]*size[2]+j*size[2]+k]
				}
			}
		}

		TensorCopyTo(host, dev)
		TensorCopyTo(hostP, devP)

		fft.Forward(dev, devT)
		TensorCopyFrom(devT, hostT)

		fftP.Forward(devP, devPT)
		TensorCopyFrom(devPT, hostPT)

		fft.Inverse(devT, devTT)
		TensorCopyFrom(devTT, hostTT)

		fftP.Inverse(devPT, devPTT)
		TensorCopyFrom(devPTT, hostPTT)

		var (
			errorTT  float = 0
			errorPTT float = 0
			errorTPT float = 0
		)
		fmt.Println("normalization:", fft.Normalization(), fftP.Normalization())
		for i := range hostTT.List() {
			hostTT.List()[i] /= float(fft.Normalization())
			if abs(host.List()[i]-hostTT.List()[i]) > errorTT {
				errorTT = abs(host.List()[i] - hostTT.List()[i])
			}
		}
		for i := range hostPTT.List() {
			hostPTT.List()[i] /= float(fftP.Normalization())
			if abs(hostP.List()[i]-hostPTT.List()[i]) > errorPTT {
				errorPTT = abs(hostP.List()[i] - hostPTT.List()[i])
			}
		}
		for i := range hostPT.List() {
			if abs(hostPT.List()[i]-hostT.List()[i]) > errorTPT {
				errorTPT = abs(hostPT.List()[i] - hostT.List()[i])
			}
		}
		//tensor.Format(os.Stdout, host2)
		fmt.Println("transformed² FFT error:                    ", errorTT)
		fmt.Println("padded+transformed² FFT error:             ", errorPTT)
		fmt.Println("transformed - padded+transformed FFT error:", errorTPT)
		if errorTT > 1E-4 || errorTPT > 1E-4 || errorPTT > 1E-4 {
			t.Fail()
		}
	}

}


func TestFFT(t *testing.T) {

	// 	for _, size := range fft_test_sizes {
	// 
	// 		paddedsize := []int{2 * size[0], 2 * size[1], 2 * size[2]}
	// 
	// 		fft := NewFFTPadded(backend, size, paddedsize)
	// 		fmt.Println(fft)
	// 		outsize := fft.PhysicSize()
	// 
	// 		devIn, devOut := NewTensor(backend, size), NewTensor(backend, outsize)
	// 		host1, host2 := tensor.NewTensorN(size), tensor.NewTensorN(size)
	// 
	// 		for i := 0; i < tensor.N(host1); i++ {
	// 			host1.List()[i] = rand.Float() //float(i%100) / 100
	// 		}
	// 
	// 		// 		host1.List()[0] = 1.
	// 
	// 
	// 		TensorCopyTo(host1, devIn)
	// 		tensor.Format(os.Stdout, devIn)
	// 		fft.Forward(devIn, devOut)
	// 		tensor.Format(os.Stdout, devOut)
	// 		fft.Inverse(devOut, devIn)
	// 		tensor.Format(os.Stdout, devIn)
	// 		TensorCopyFrom(devIn, host2)
	// 
	// 		N := float(fft.Normalization())
	// 		var maxError float = 0
	// 		for i := range host2.List() {
	// 			host2.List()[i] /= N
	// 			if abs(host2.List()[i]-host1.List()[i]) > maxError {
	// 				maxError = abs(host2.List()[i] - host1.List()[i])
	// 			}
	// 		}
	// 		//tensor.Format(os.Stdout, host2)
	// 		fmt.Println("FFT error:", maxError)
	// 		if maxError > 1E-4 {
	// 			t.Fail()
	// 		}
	// 	}

}


// func abs(r float) float {
// 	if r < 0 {
// 		return -r
// 	}
// 	//else
// 	return r
// }
