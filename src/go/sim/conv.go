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
	kernel [6]*DevTensor
	buffer [3]*DevTensor
	mcomp  [3]*DevTensor // only a buffer, automatically set at each conv()
	hcomp  [3]*DevTensor // only a buffer, automatically set at each conv()
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
		conv.buffer[i] = NewTensor(conv.Backend, conv.PhysicSize())
		conv.mcomp[i] = &DevTensor{conv.Backend, dataSize, unsafe.Pointer(nil)}
		conv.hcomp[i] = &DevTensor{conv.Backend, dataSize, unsafe.Pointer(nil)}
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

	// initialize mcomp, hcomp, re-using them from conv to avoid repeated allocation
	mcomp, hcomp := conv.mcomp, conv.hcomp
	buffer := conv.buffer
	kernel := conv.kernel
	mLen := Len(mcomp[0].size)
	for i := 0; i < 3; i++ {
		mcomp[i].data = conv.arrayOffset(source.data, i*mLen)
		hcomp[i].data = conv.arrayOffset(dest.data, i*mLen)
	}

	//Sync

	// Forward FFT
	for i := 0; i < 3; i++ {
		conv.Forward(mcomp[i], buffer[i]) // should not be asynchronous unless we have 3 fft's (?)
	}

	// Point-wise kernel multiplication in reciprocal space
	conv.kernelMul(buffer[X].data, buffer[Y].data, buffer[Z].data,
		kernel[XX].data, kernel[YY].data, kernel[ZZ].data,
		kernel[YZ].data, kernel[XZ].data, kernel[XY].data,
		6, Len(buffer[X].size)) // nRealNumbers

	// Inverse FFT
	for i := 0; i < 3; i++ {
		conv.Inverse(buffer[i], hcomp[i]) // should not be asynchronous unless we have 3 fft's (?)
	}
}

// INTERNAL: Loads a convolution kernel.
// This is automatically done during initialization.
func (conv *Conv) loadKernel6(kernel []*tensor.Tensor3) {

	for _, k := range kernel {
		if k != nil {
			assert(tensor.EqualSize(k.Size(), conv.KernelSize()))
		}
	}

	buffer := tensor.NewTensorN(conv.KernelSize())
	devbuf := NewTensor(conv.Backend, conv.KernelSize())

	fft := NewFFT(conv.Backend, conv.KernelSize())
	N := 1.0 / float(fft.Normalization())

	for i := range conv.kernel {
		if kernel[i] != nil { // nil means it would contain only zeros so we don't store it.
			conv.kernel[i] = NewTensor(conv.Backend, conv.PhysicSize())
			tensor.CopyTo(kernel[i], buffer)
			for i := range buffer.List() {
				buffer.List()[i] *= N
			}
			ZeroTensor(conv.kernel[i])
			TensorCopyTo(buffer, devbuf)
			fft.Forward(devbuf, conv.kernel[i])
		}
	}
}


// size of magnetization + padding zeros, this is the FFT logicSize
func (conv *Conv) KernelSize() []int {
	return conv.LogicSize()
}
