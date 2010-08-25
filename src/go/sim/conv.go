package sim

import (
	"tensor"
	"unsafe"
)


type Conv struct {
	FFT
	kernel [6]*Tensor
	buffer [3]*Tensor
	mcomp  [3]*Tensor // only a buffer, automatically set at each conv()
	hcomp  [3]*Tensor // only a buffer, automatically set at each conv()
}


func NewConv(backend Backend, dataSize []int, kernel []tensor.StoredTensor) *Conv {
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
		conv.mcomp[i] = &Tensor{conv.Backend, dataSize, unsafe.Pointer(nil)}
		conv.hcomp[i] = &Tensor{conv.Backend, dataSize, unsafe.Pointer(nil)}
	}
	conv.LoadKernel6(kernel)

	return conv
}


func (conv *Conv) Convolve(source, dest *Tensor) {
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

	for i := 0; i < 3; i++ {
		ZeroTensor(buffer[i])
		CopyPad(mcomp[i], buffer[i])
		//     fmt.Println("mPadded", i)
		//     tensor.Format(os.Stdout, buffer[i])
	}

	//Sync

	for i := 0; i < 3; i++ {
		conv.Forward(buffer[i], buffer[i]) // should not be asynchronous unless we have 3 fft's (?)
		//     fmt.Println("fftm", i)
		//     tensor.Format(os.Stdout, buffer[i])
	}

	conv.kernelMul(buffer[X].data, buffer[Y].data, buffer[Z].data,
		kernel[XX].data, kernel[YY].data, kernel[ZZ].data,
		kernel[YZ].data, kernel[XZ].data, kernel[XY].data,
		6, Len(buffer[X].size)) // nRealNumbers

	//   for i:=0; i<3; i++{
	//     fmt.Println("mulM", i)
	//     tensor.Format(os.Stdout, buffer[i])
	//   }

	for i := 0; i < 3; i++ {
		conv.Inverse(buffer[i], buffer[i]) // should not be asynchronous unless we have 3 fft's (?)
	}

	for i := 0; i < 3; i++ {
		CopyUnpad(buffer[i], hcomp[i])
	}
}


func (conv *Conv) LoadKernel6(kernel []tensor.StoredTensor) {
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

			ZeroTensor(devbuf)
			ZeroTensor(conv.kernel[i])
			TensorCopyTo(buffer, devbuf)
			CopyPad(devbuf, conv.kernel[i]) ///@todo padding should be done on host, not device, to save sim memory / avoid fragmentation

			fft.Forward(conv.kernel[i], conv.kernel[i])
		}
	}
}


/// size of the magnetization and field, this is the FFT dataSize
// func (conv *Conv) DataSize() []int{
//   return conv.fft.DataSize()
// }


/// size of magnetization + padding zeros, this is the FFT logicSize /// todo remove in favor of embedded LogicSize()
func (conv *Conv) KernelSize() []int {
	return conv.LogicSize()
}


/// size of magnetization + padding zeros + striding zeros, this is the FFT logicSize
// func (conv *Conv) PhysicSize() []int{
//   return conv.fft.PhysicSize()
// }

const (
	XX = 0
	YY = 1
	ZZ = 2
	YZ = 3
	XZ = 4
	XY = 5
)
