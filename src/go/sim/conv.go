//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import (
	. "mumax/common"
	"mumax/tensor"
	"fmt"
	"os"
)

// "Conv" is a 3D vector convolution "plan".
// It convolutes (mx, my, mz) (r) with a symmetric kernel
// (Kxx Kxy Kxz)
// (Kyx Kyy Kyz) (r)
// (Kzx Kzy Kzz)
// This is the convolution needed for calculating the magnetostatic field.
// If the convolution kernel is larger than the input data, the extra
// space is padded with zero's (which are efficiently handled).
type Conv struct {
	FFT
	kernel [6]*DevTensor
	buffer [3]*DevTensor
}


// dataSize = size of input data (one componenten of the magnetization), e.g., 4 x 32 x 32.
// The size of the kernel componenents (Kxx, Kxy, ...) must be at least the size of the input data,
// but may be larger. Typically, there will be zero-padding by a factor of 2. e.g. the kernel
// size may be 8 x 64 x 64.
func NewConv(backend *Backend, dataSize []int, kernel []*tensor.T3) *Conv {
	// size checks
	kernelSize := kernel[XX].Size()
	assert(len(dataSize) == 3)
	assert(len(kernelSize) == 3)
	for i := range dataSize {
		Assert(dataSize[i] <= kernelSize[i])
	}

	conv := new(Conv)
	conv.FFT = *NewFFTPadded(backend, dataSize, kernelSize)

	///@todo do not allocate for infinite2D problem
	for i := 0; i < 3; i++ {
		conv.buffer[i] = NewTensor(conv.Backend, conv.PhysicSize())
	}
	conv.loadKernel6(kernel)

	return conv
}


func (conv *Conv) Convolve(source, dest *DevTensor) {
	Debugvv("Conv.Convolve()")
	assert(len(source.size) == 4) // size checks
	assert(len(dest.size) == 4)
	for i, s := range conv.DataSize() {
		assert(source.size[i+1] == s)
		assert(dest.size[i+1] == s)
	}

	mcomp, hcomp := source.comp, dest.comp
	buffer := conv.buffer
	kernel := conv.kernel

	//Sync

	// Forward FFT
	for i := 0; i < 3; i++ {
		conv.Forward(mcomp[i], buffer[i]) // should not be asynchronous unless we have 3 fft's (?)
	}

	// Point-wise kernel multiplication in reciprocal space
	kernType := conv.KernType()

	switch kernType {
	default:
		panic("Bug")
	case 6:
		conv.kernelMul(buffer[X].data, buffer[Y].data, buffer[Z].data,
			kernel[XX].data, kernel[YY].data, kernel[ZZ].data,
			kernel[YZ].data, kernel[XZ].data, kernel[XY].data,
			kernType, Len(buffer[X].size)) // nRealNumbers
	case 4:
		conv.kernelMul(buffer[X].data, buffer[Y].data, buffer[Z].data,
			kernel[XX].data, kernel[YY].data, kernel[ZZ].data,
			kernel[YZ].data, 0, 0,
			kernType, Len(buffer[X].size)) // nRealNumbers
	}
	//conv.Stop()

	// Inverse FFT
	for i := 0; i < 3; i++ {
		conv.Inverse(buffer[i], hcomp[i]) // should not be asynchronous unless we have 3 fft's (?)
	}
}

// INTERNAL: Loads a convolution kernel.
// This is automatically done during initialization.
// "kernel" is not FFT'ed yet, this is done here.
// We use exactly the same fft as for the magnetizaion
// so that the convolution definitely works.
// After FFT'ing, the kernel is purely real,
// so we discard the imaginary parts.
// This saves a huge amount of memory
func (conv *Conv) loadKernel6(kernel []*tensor.T3) {

	// Check sanity of kernel
	for i, k := range kernel {
		if k != nil && conv.needKernComp(i) {
			Println("Need Kernel component " + KernString[i])
			Assert(tensor.EqualSize(k.Size(), conv.LogicSize()))
			for _, e := range k.List() {
				if !IsReal(e) {
					tensor.Format(os.Stderr, k)
				}
				AssertMsg(IsReal(e), "K", KernString[i], " is NaN or Inf") // should not be NaN or Inf
			}
		}
	}

	fft := NewFFT(conv.Backend, conv.LogicSize())
	norm := 1.0 / float32(fft.Normalization())
	devIn := NewTensor(conv.Backend, conv.LogicSize())
	devOut := NewTensor(conv.Backend, fft.PhysicSize())
	hostOut := tensor.NewT3(fft.PhysicSize())

	//   allocCount := 0

	for i := range conv.kernel {
		if conv.needKernComp(i) { // the zero components are not stored
			//       allocCount++
			TensorCopyTo(kernel[i], devIn)
			fft.Forward(devIn, devOut)
			TensorCopyFrom(devOut, hostOut)
			listOut := hostOut.List()

			// Normally, the FFT'ed kernel is purely real because of symmetry,
			// so we only store the real parts...
			maximg := float32(0.)
			for j := 0; j < len(listOut)/2; j++ {
				listOut[j] = listOut[2*j] * norm
				if abs32(listOut[2*j+1]) > maximg {
					maximg = abs32(listOut[2*j+1])
				}
			}
			// ...however, we check that the imaginary parts are nearly zero,
			// just to be sure we did not make a mistake during kernel creation.
			if maximg > 1e-4 {
				fmt.Fprintln(os.Stderr, "Warning: FFT Kernel max imaginary part=", maximg)
			}

			conv.kernel[i] = NewTensor(conv.Backend, conv.KernelSize())
			conv.memcpyTo(&listOut[0], conv.kernel[i].data, Len(conv.kernel[i].Size()))
		}
	}

	//   fmt.Println(allocCount, " non-zero kernel components.")
	fft.Free()
	devIn.Free()
	devOut.Free()

}

// The kernel type: how many kernel components are nonzero.
// General 3D case: 6 (symmetric kernel)
// 2D case: 4 (Kxy = KxZ = 0)
// infinitely thick 2D case: 3 (Kxy = KxZ = Kxx = 0)
func (conv *Conv) KernType() int {
	kernType := 6
	if conv.logicSize[X] == 1 { // 2D simulation: Kxy = Kxz = 0, do not multiply with zeros
		kernType = 4
	}
	return kernType
}

// Returns true if a a kernel component (e.g. XX) is non-zero
func (conv *Conv) needKernComp(comp int) bool {
	if conv.KernType() == 6 {
		return true
	}
	if comp == XY || comp == XZ {
		return false
	}
	if conv.KernType() == 3 && comp == XX {
		return false
	}
	return true
}

// size of the (real) kernel
func (conv *Conv) KernelSize() []int {
	return []int{conv.PhysicSize()[X], conv.PhysicSize()[Y], conv.PhysicSize()[Z] / 2}
}
