//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import (
	. "mumax/common"
	"mumax/tensor"
	"unsafe"
	//"fmt"
)

/**
 * A Backend wraps some unsafe methods of the Device interface
 * with safe versions, and provides additional methods derived
 * from low-end Devices methods. Therefore, the user should
 * interact with a Backend and forget about the underlying Device.
 *
 * Other unsafe methods are wrapped by higher-lever structs that
 * embed a Backend. E.g. FFT wraps the fft functions, Conv wraps
 * convolution functions, etc.
 *
 */

type Backend struct {
	Device
	gpuid int
	Timer
}

func NewBackend(d Device) *Backend {
	return &Backend{d, -1, NewTimer()}
}

//_________________________________________________________________________ safe wrappers for Device methods

// more or less safe initialization, calls the underlying init() only once
// (given you allocate only one unique CPU, GPU, ...)
// func (dev *Backend) InitBackend() {
// 	if !dev.Initiated {
// 		dev.init()
// 		dev.Initiated = true
// 	}
// }

func (dev *Backend) SetDevice(gpuid int) {
	if dev.gpuid != gpuid {
		dev.gpuid = gpuid
		dev.setDevice(gpuid)
	}
}

// Copies a number of float32s from host to GPU
func (dev *Backend) memcpyTo(source *float32, dest uintptr, nFloats int) {
	dev.memcpy(uintptr(unsafe.Pointer(source)), dest, nFloats, CPY_TO)
}

// Copies a number of float32s from GPU to host
func (dev *Backend) memcpyFrom(source uintptr, dest *float32, nFloats int) {
	dev.memcpy(source, uintptr(unsafe.Pointer(dest)), nFloats, CPY_FROM)
}

// Copies a number of float32s from GPU to GPU
func (dev *Backend) memcpyOn(source, dest uintptr, nFloats int) {
	dev.memcpy(source, dest, nFloats, CPY_ON)
}

// Copies from a smaller to a larger tensor, not touching the additional space in the destination (typically filled with zero padding)
func (dev *Backend) copyPad(source, dest uintptr, sourceSize, destSize []int) {
	dev.copyPadded(source, dest, sourceSize, destSize, CPY_PAD)
}

//Copies from a larger to a smaller tensor, not reading the additional data in the source (typically filled with zero padding or spoiled data)
func (dev *Backend) copyUnpad(source, dest uintptr, sourceSize, destSize []int) {
	dev.copyPadded(source, dest, sourceSize, destSize, CPY_UNPAD)
}

// // Gets one float32 from a Device array.
// // Slow, for debug only
// func (dev *Backend) arrayGet(array uintptr, index int) float32 {
// 	var f float32
// 	dev.memcpyFrom(dev.arrayOffset(array, index), &f, 1)
// 	return f
// }
// 
// // Sets one float32 on a Device array.
// // Slow, for debug only
// func (dev *Backend) arraySet(array uintptr, index int, value float32) {
// 	dev.memcpyTo(&value, dev.arrayOffset(array, index), 1)
// }


// a[i] += b[i]
func (dev *Backend) Add(a, b *DevTensor) {
	assert(tensor.EqualSize(a.size, b.size))
	dev.add(a.data, b.data, tensor.Prod(a.Size()))
}

// a[i] += cnst * b[i]
func (dev *Backend) MAdd(a *DevTensor, cnst float32, b *DevTensor) {
	assert(tensor.EqualSize(a.size, b.size))
	//fmt.Println("madd ", a, b)
	//adata := a.data
	//bdata := b.data
	//asize :=  tensor.Prod(a.Size())
	dev.madd(a.data, cnst, b.data, a.length)
}

// a[i] += b[i] * c[i]
func (dev *Backend) MAdd2(a, b, c *DevTensor) {
	assert(tensor.EqualSize(a.size, b.size))
	assert(tensor.EqualSize(b.size, c.size))
	dev.madd2(a.data, b.data, c.data, tensor.Prod(a.Size()))
}

func(dev *Backend) ScaledDotProduct(result, a, b *DevTensor, scale float32){
	assert(tensor.EqualSize(a.size, b.size))
	assert(tensor.EqualSize(a.size, result.size))
	dev.scaledDotProduct(result.data, a.data, b.data, scale, tensor.Prod(a.size))
}

func (dev *Backend) AddLinAnis(h, m *DevTensor, K []*DevTensor) {
	dev.addLinAnis(h.comp[X].data, h.comp[Y].data, h.comp[Z].data,
		m.comp[X].data, m.comp[Y].data, m.comp[Z].data,
		K[XX].data, K[YY].data, K[ZZ].data,
		K[YZ].data, K[XZ].data, K[XY].data,
		h.length)
}


// a[i]  = weightA * a[i] + weightB * b[i]
func (dev *Backend) LinearCombination(a, b *DevTensor, weightA, weightB float32) {
	assert(tensor.EqualSize(a.size, b.size))
	dev.linearCombination(a.data, b.data, weightA, weightB, tensor.Prod(a.Size()))
}

// a[i] += cnst
func (dev *Backend) AddConstant(a *DevTensor, cnst float32) {
	dev.addConstant(a.data, cnst, tensor.Prod(a.Size()))
}

// func (dev *Backend) Normalize(m *DevTensor) {
// 	assert(len(m.size) == 4)
// 	N := m.size[1] * m.size[2] * m.size[3]
// 	dev.normalize(m.data, N)
// }

func (dev *Backend) AddLocalFields(m, h *DevTensor, hext []float32, anisType int, anisK, anisAxes []float32) {
	assert(m.length == h.length)
	assert(len(m.size) == 4)
	//fmt.Printf("hext:%v, anistype:%v, anisK:%v, anisAxes:%v\n", hext, anisType, anisK, anisAxes)
	dev.addLocalFields(m.data, h.data, hext, anisType, anisK, anisAxes, m.length/3)
}


func (dev *Backend) SemianalStep(min, mout, h *DevTensor, dt, alpha float32) {
	assert(min.length == h.length)
	assert(min.length == mout.length)
	dev.semianalStep(min.data, mout.data, h.data, dt, alpha, min.length/3)
}

func (b Backend) OverrideStride(stride int) {
	panic("OverrideStride is currently not compatible with the used FFT, it should always be 1")
	Debugvv("Backend.OverrideStride(", stride, ")")
	assert(stride > 0 || stride == -1)
	b.overrideStride(stride)
}

// unsafe FFT
func (b Backend) fftForward(plan uintptr, in, out uintptr) {
	b.fft(plan, in, out, FFT_FORWARD)
}

// unsafe FFT
func (b Backend) fftInverse(plan uintptr, in, out uintptr) {
	b.fft(plan, in, out, FFT_INVERSE)
}

// func (b Backend) ExtractReal(complex, real *Tensor) {
// 	assert(Len(complex.size) == 2*Len(real.size))
// 	b.extractReal(complex.data, real.data, Len(real.size))
// }

//________________________________________________________________________ derived methods


// Takes an array size and returns the smallest multiple of Stride() where the array size fits in
func (b Backend) PadToStride(nFloats int) int {
	stride := b.Stride()
	gpulen := ((nFloats-1)/stride + 1) * stride

	assert(gpulen%stride == 0)
	assert(gpulen > 0)
	assert(gpulen >= nFloats)
	return gpulen
}
