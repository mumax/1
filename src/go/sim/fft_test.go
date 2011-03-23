//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import (
	"testing"
	"mumax/tensor"
	"fmt"
	"rand"
)

var backend = GPU

var fft_test_sizes [][]int = [][]int{
  {1,16,16}}

func TestTimeFFT(t *testing.T) {
// func testFFTPadded2(t *testing.T) {
  
var basic_size []int = []int{1,32,32}
var size []int = []int{1,16,16}


  for i := 1; i < 11; i++ {
    size[0] = basic_size[0]
    size[1] = i*basic_size[1]
    size[2] = i*basic_size[2]

    fmt.Println("cnt: ", i, "Size in: ", size)
    fmt.Println()
    fmt.Println()
    fmt.Println()
    fmt.Println()
    fmt.Println()
    paddedsize := padSize(size, []int{0, 0, 0})

    fft := NewFFTPadded(backend, size, paddedsize)
    fftP := NewFFT(backend, paddedsize) // with manual padding

    fmt.Println(fft)
    fmt.Println(fftP)

    outsize := fftP.PhysicSize()

    host := tensor.NewT(size)
    dev, devT, devTT := NewTensor(backend, size), NewTensor(backend, outsize), NewTensor(backend, size)

    host.List()[0] = 1.
    for i := 0; i < size[0]; i++ {
      for j := 0; j < size[1]; j++ {
        for k := 0; k < size[2]; k++ {
          host.List()[i*size[1]*size[2]+j*size[2]+k] = rand.Float32() //1.
        }
      }
    }


    TensorCopyTo(host, dev)

    for j := 1; j < 10; j++ {
      fft.Forward(dev, devT)
      fft.Inverse(devT, devTT)
    }

    fft.Free()
    fftP.Free()
    
    dev.Free()
    devT.Free()
    devTT.Free()
    
  }


}


func TestFFTPadded(t *testing.T) {


	for _, size := range fft_test_sizes {
		fmt.Println("Size: ", size)
		paddedsize := padSize(size, []int{0, 0, 0})

		fft := NewFFTPadded(backend, size, paddedsize)
		fftP := NewFFT(backend, paddedsize) // with manual padding

		fmt.Println(fft)
		fmt.Println(fftP)

		outsize := fftP.PhysicSize()

		dev, devT, devTT := NewTensor(backend, size), NewTensor(backend, outsize), NewTensor(backend, size)
		devP, devPT, devPTT := NewTensor(backend, paddedsize), NewTensor(backend, outsize), NewTensor(backend, paddedsize)

		host, hostT, hostTT := tensor.NewT(size), tensor.NewT(outsize), tensor.NewT(size)
		hostP, hostPT, hostPTT := tensor.NewT(paddedsize), tensor.NewT(outsize), tensor.NewT(paddedsize)

		host.List()[0] = 1.
		for i := 0; i < size[0]; i++ {
			for j := 0; j < size[1]; j++ {
				for k := 0; k < size[2]; k++ {
					host.List()[i*size[1]*size[2]+j*size[2]+k] = rand.Float32() //1.
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

		fmt.Println("in:")
		//host.WriteTo(os.Stdout)

		fmt.Println("out(padded):")
		//hostT.WriteTo(os.Stdout)

		fmt.Println("backtransformed:")
		//hostTT.WriteTo(os.Stdout)

		var (
			errorTT  float32 = 0
			errorPTT float32 = 0
			errorTPT float32 = 0
		)
		fmt.Println("normalization:", fft.Normalization(), fftP.Normalization())
		for i := range hostTT.List() {
			hostTT.List()[i] /= float32(fft.Normalization())
			if abs(host.List()[i]-hostTT.List()[i]) > errorTT {
				errorTT = abs(host.List()[i] - hostTT.List()[i])
			}
		}
		for i := range hostPTT.List() {
			hostPTT.List()[i] /= float32(fftP.Normalization())
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

	for _, size := range fft_test_sizes {
		fmt.Println("Size: ", size)
		fft := NewFFT(backend, size)
		fmt.Println(fft)
		outsize := fft.PhysicSize()

		dev, devT, devTT := NewTensor(backend, size), NewTensor(backend, outsize), NewTensor(backend, size)

		host, hostT, hostTT := tensor.NewT(size), tensor.NewT(outsize), tensor.NewT(size)

		for i := 0; i < size[0]; i++ {
			for j := 0; j < size[1]; j++ {
				for k := 0; k < size[2]; k++ {
					host.List()[i*size[1]*size[2]+j*size[2]+k] = 0. + 1.*(rand.Float32()-.5) //1.
				}
			}
		}
		//     host.List()[63] = 1.

		//     list := host.List()
		//     for i:= range list{
		//       list[i]=float32(i)
		//     }

		TensorCopyTo(host, dev)

		fft.Forward(dev, devT)
		TensorCopyFrom(devT, hostT)

		fft.Inverse(devT, devTT)
		TensorCopyFrom(devTT, hostTT)

		fmt.Println("in:")
		//host.WriteTo(os.Stdout)

		fmt.Println("out:")
		//hostT.WriteTo(os.Stdout)

		fmt.Println("backtransformed:")
		//hostTT.WriteTo(os.Stdout)

		var (
			errorTT float32 = 0
		)

		fmt.Println("Normalization: ", fft.Normalization())
		for i := range hostTT.List() {
			hostTT.List()[i] /= float32(fft.Normalization())
			if abs(host.List()[i]-hostTT.List()[i]) > errorTT {
				errorTT = abs(host.List()[i] - hostTT.List()[i])
			}
		}
		//tensor.Format(os.Stdout, host2)
		fmt.Println("transformed² FFT error:                    ", errorTT)
		if errorTT > 1E-4 {
			t.Fail()
		}
	}

}


func abs(r float32) float32 {
	if r < 0 {
		return -r
	}
	//else
	return r
}
