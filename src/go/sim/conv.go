package sim

import (
	"tensor"
	"unsafe"
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
	kernel [6]*Tensor
	fftm [3]*Tensor
	mcomp  [3]*Tensor // only a fftm, automatically set at each conv()
	hcomp  [3]*Tensor // only a fftm, automatically set at each conv()
}

// dataSize = size of input data (one componenten of the magnetization), e.g., 4 x 32 x 32.
// The size of the kernel componenents (Kxx, Kxy, ...) must be at least the size of the input data,
// but may be larger. Typically, there will be zero-padding by a factor of 2. e.g. the kernel
// size may be 8 x 64 x 64.
func NewConv(backend *Backend, dataSize []int, kernel []*tensor.Tensor3) *Conv {
	// size checks
	kernelSize := kernel[XX].Size()
	assert(len(dataSize) == 3)
	assert(len(kernelSize) == 3)
	for i := range dataSize {
		assert(dataSize[i] <= kernelSize[i])
	}

	conv := new(Conv)
	conv.FFT = *NewFFTPadded(backend, dataSize, kernelSize)

	///@todo do not allocate for infinite2D problem
	for i := 0; i < 3; i++ {
		conv.fftm[i] = NewTensor(conv.Backend, conv.PhysicSize())
		conv.mcomp[i] = &Tensor{conv.Backend, dataSize, unsafe.Pointer(nil)}
		conv.hcomp[i] = &Tensor{conv.Backend, dataSize, unsafe.Pointer(nil)}
	}
	conv.loadKernel6(kernel)

	return conv
}


func (conv *Conv) Convolve(source, dest *Tensor) {
	//Debugvv("Conv.Convolve()")
	assert(len(source.size) == 4) // size checks
	assert(len(dest.size) == 4)
	for i, s := range conv.DataSize() {
		assert(source.size[i+1] == s)
		assert(dest.size[i+1] == s)
	}

	// initialize mcomp, hcomp, re-using them from conv to avoid repeated allocation
	mcomp, hcomp := conv.mcomp, conv.hcomp
	fftm := conv.fftm
	kernel := conv.kernel
	mLen := Len(mcomp[0].size)
	for i := 0; i < 3; i++ {
		mcomp[i].data = conv.arrayOffset(source.data, i*mLen)
		hcomp[i].data = conv.arrayOffset(dest.data, i*mLen)
	}

	//Sync

	// Forward FFT
  conv.Start("FFT")
	for i := 0; i < 3; i++ {
		conv.Forward(mcomp[i], fftm[i]) // should not be asynchronous unless we have 3 fft's (?)
	}
	conv.Stop("FFT")

	// Point-wise kernel multiplication in reciprocal space
  conv.Start("Kernel multiplication")
	conv.kernelMul(fftm[X].data, fftm[Y].data, fftm[Z].data,
		kernel[XX].data, kernel[YY].data, kernel[ZZ].data,
		kernel[YZ].data, kernel[XZ].data, kernel[XY].data,
		6, Len(fftm[X].size)) // nRealNumbers
  conv.Stop("Kernel multiplication")
 
	// Inverse FFT
  conv.Start("FFT")
	for i := 0; i < 3; i++ {
		conv.Inverse(fftm[i], hcomp[i]) // should not be asynchronous unless we have 3 fft's (?)
	}
	conv.Stop("FFT")
}

// INTERNAL: Loads a convolution kernel.
// This is automatically done during initialization.
func (conv *Conv) loadKernel6(kernel []*tensor.Tensor3) {

	for _, k := range kernel {
		if k != nil {
			assert(tensor.EqualSize(k.Size(), conv.KernelSize()))
		}
	}

	fftm := tensor.NewTensorN(conv.KernelSize())
	devbuf := NewTensor(conv.Backend, conv.KernelSize())

	fft := NewFFT(conv.Backend, conv.KernelSize())
	N := 1.0 / float(fft.Normalization())

	for i := range conv.kernel {
		if kernel[i] != nil { // nil means it would contain only zeros so we don't store it.
			conv.kernel[i] = NewTensor(conv.Backend, conv.PhysicSize())
			tensor.CopyTo(kernel[i], fftm)
			for i := range fftm.List() {
				fftm.List()[i] *= N
			}
			ZeroTensor(conv.kernel[i])
			TensorCopyTo(fftm, devbuf)
			fft.Forward(devbuf, conv.kernel[i])
		}
	}
}


// size of magnetization + padding zeros, this is the FFT logicSize
func (conv *Conv) KernelSize() []int {
	return conv.LogicSize()
}
