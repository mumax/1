//  This file is part of MuMax, a high-perfomrance micromagnetic simulator.
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package ce

import (
	"mumax"
)

// The abstracted ce.
// Can be, e.g., a GPU, CPU, multi-GPU,
// or any "ce" that implements ce.Interface.
var ce Interface

// Initializes the ce to use GPU number "gpu_id",
// with maximum "threads" threads per thread block.
// The options flag is currently not used. 
func UseGpu(gpu_id, threads, options int){
	if ce != nil{
		panic(mumax.Bug("device allready set"))
	} else {
		ce = Gpu{}
		ce.setDevice(gpu_id)
		ce.init(threads, options)
	}
}


// Copies a number of float32s from host to GPU
func memcpyTo(source *float32, dest uintptr, nFloats int) {
	device.memcpy(uintptr(unsafe.Pointer(source)), dest, nFloats, CPY_TO)
}

// Copies a number of float32s from GPU to host
func memcpyFrom(source uintptr, dest *float32, nFloats int) {
	device.memcpy(source, uintptr(unsafe.Pointer(dest)), nFloats, CPY_FROM)
}

// Copies a number of float32s from GPU to GPU
func memcpyOn(source, dest uintptr, nFloats int) {
device.memcpy(source, dest, nFloats, CPY_ON)
}

// Copies from a smaller to a larger tensor, not touching the additional space in the destination (typically filled with zero padding)
// TODO: check if still needed?
func copyPad(source, dest uintptr, sourceSize, destSize []int) {
	device.copyPadded(source, dest, sourceSize, destSize, CPY_PAD)
}

//Copies from a larger to a smaller tensor, not reading the additional data in the source (typically filled with zero padding or spoiled data)
// TODO: check if still needed?
func copyUnpad(source, dest uintptr, sourceSize, destSize []int) {
	device.copyPadded(source, dest, sourceSize, destSize, CPY_UNPAD)
}


// a[i] += b[i]
func Add(a, b *Tensor) {
	assert(tensor.EqualSize(a.size, b.size))
	device.add(a.data, b.data, tensor.Prod(a.Size()))
}

// a[i] += cnst * b[i]
func MAdd(a *Tensor, cnst float32, b *Tensor) {
	assert(tensor.EqualSize(a.size, b.size))
	device.madd(a.data, cnst, b.data, a.length)
}

// a[i] += b[i] * c[i]
func MAdd2(a, b, c *Tensor) {
	assert(tensor.EqualSize(a.size, b.size))
	assert(tensor.EqualSize(b.size, c.size))
	device.madd2(a.data, b.data, c.data, tensor.Prod(a.Size()))
}

func AddLinAnis(h, m *Tensor, K []*Tensor) {
	device.addLinAnis(h.comp[X].data, h.comp[Y].data, h.comp[Z].data,
		m.comp[X].data, m.comp[Y].data, m.comp[Z].data,
		K[XX].data, K[YY].data, K[ZZ].data,
		K[YZ].data, K[XZ].data, K[XY].data,
		h.length)
}

// a[i]  = weightA * a[i] + weightB * b[i]
func (* LinearCombination(a, b *Tensor, weightA, weightB float32) {
	assert(tensor.EqualSize(a.size, b.size))
	device.linearCombination(a.data, b.data, weightA, weightB, tensor.Prod(a.Size()))
}

// a[i] += cnst
func (* AddConstant(a *Tensor, cnst float32) {
	addConstant(a.data, cnst, tensor.Prod(a.Size()))
}

// func (* Normalize(m *Tensor) {
// 	assert(len(m.size) == 4)
// 	N := m.size[1] * m.size[2] * m.size[3]
// 	normalize(m.data, N)
// }

func (* AddLocalFields(m, h *Tensor, hext []float32, anisType int, anisK, anisAxes []float32) {
	assert(m.length == h.length)
	assert(len(m.size) == 4)
	//fmt.Printf("hext:%v, anistype:%v, anisK:%v, anisAxes:%v\n", hext, anisType, anisK, anisAxes)
	addLocalFields(m.data, h.data, hext, anisType, anisK, anisAxes, m.length/3)
}


func (* SemianalStep(min, mout, h *Tensor, dt, alpha float32) {
	assert(min.length == h.length)
	assert(min.length == mout.length)
	semianalStep(min.data, mout.data, h.data, dt, alpha, min.length/3)
}

func (b  OverrideStride(stride int) {
	panic("OverrideStride is currently not compatible with the used FFT, it should always be 1")
	Debugvv("Backend.OverrideStride(", stride, ")")
	assert(stride > 0 || stride == -1)
	b.overrideStride(stride)
}

// unsafe FFT
func (b  fftForward(plan uintptr, in, out uintptr) {
	b.fft(plan, in, out, FFT_FORWARD)
}

// unsafe FFT
func (b  fftInverse(plan uintptr, in, out uintptr) {
	b.fft(plan, in, out, FFT_INVERSE)
}

// func (b  ExtractReal(complex, real *Tensor) {
// 	assert(Len(complex.size) == 2*Len(real.size))
// 	b.extractReal(complex.data, real.data, Len(real.size))
// }

//________________________________________________________________________ derived methods


// Takes an array size and returns the smallest multiple of Stride() where the array size fits in
func (b  PadToStride(nFloats int) int {
	stride := b.Stride()
	gpulen := ((nFloats-1)/stride + 1) * stride

	assert(gpulen%stride == 0)
	assert(gpulen > 0)
	assert(gpulen >= nFloats)
	return gpulen
}


// Panics if test is false
func assert(test bool){
	if !test{
		panic(mumax.Bug("Assertion failed."))
	}
}

func assertEqual(a, b int){
	if a != b{
		panic(mumax.Bug("Assertion failed."))
	}
}

func assertEqualSize(a, b []int){
	if len(a) != len(b){
		panic(mumax.Bug("Assertion failed."))
	}
	for i,a_i := range a{
		if a_i != b[i]{
		panic(mumax.Bug("Assertion failed."))
		}
	}
}
