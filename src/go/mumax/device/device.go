//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.


// The device package implements the communication with a high-performance computing device
// like a GPU, CPU, multi-GPU or any "device" that implements device.Interface.
//
// The device is fully abstracted: CPU's, GPU's, ... are all represented by the same interface.
// Data on a a device does not live in the Go memory space but is represented
// by a device.Tensor that references to data on the device.
// Typically, data is copied to the device, then functions are called on it
// and the result is copied back to Go memory space.
package device


import (
	. "mumax/common"
)


// Set a device for global use.
// Can be done only once.
func Use(device Interface) {
	if dev != nil {
		panic(Bug("device.Use(): device already set."))
	} else {
		dev = device
	}
}


// INTERNAL:
// Global variable that points to the abstracted device.
// 
// In a previous implementation, this was not global and
// every type that used a device needed to embed a pointer
// to it. However, there appeared to be absolutely no need
// to use different devices in the same program and this only
// gave initialization difficulties. So one global device
// seems more suited. If desired, a local device.Interface
// variable can still be used to represent a different device. 
var dev Interface


// Copies nFloats to, on or from the device, depending on the direction flag (1, 2 or 3)
//func Memcpy(source, dest uintptr, nFloats, direction int){
//	device.Memcpy
//}

// Offset the array pointer by "index" float32s, useful for taking sub-arrays
// TODO: on a multi-device this will not work
//arrayOffset(array uintptr, index int) uintptr

// Overwrite n float32s with zeros
func Zero(data uintptr, nFloats int)
// Copies a number of float32s from host to GPU
//func memcpyTo(source *float32, dest uintptr, nFloats int) {
//	device.memcpy(uintptr(unsafe.Pointer(source)), dest, nFloats, CPY_TO)
//}
//
//// Copies a number of float32s from GPU to host
//func memcpyFrom(source uintptr, dest *float32, nFloats int) {
//	device.memcpy(source, uintptr(unsafe.Pointer(dest)), nFloats, CPY_FROM)
//}
//
//// Copies a number of float32s from GPU to GPU
//func memcpyOn(source, dest uintptr, nFloats int) {
//	device.memcpy(source, dest, nFloats, CPY_ON)
//}


// Copies from a smaller to a larger tensor, not touching the additional space in the destination (typically filled with zero padding)
func CopyPad(source, dest *Tensor) {
	dev.CopyPadded(source.data, dest.data, source.size, dest.size, CPY_PAD)
}

//Copies from a larger to a smaller tensor, not reading the additional data in the source (typically filled with zero padding or spoiled data)
func CopyUnpad(source, dest *Tensor) {
	dev.CopyPadded(source.data, dest.data, source.size, dest.size, CPY_UNPAD)
}


// a[i] += b[i]
func Add(a, b *Tensor) {
	AssertEqual(a.size, b.size)
	dev.Add(a.data, b.data, a.length)
}

// a[i] += cnst * b[i]
func MAdd(a *Tensor, cnst float32, b *Tensor) {
	AssertEqual(a.size, b.size)
	dev.Madd(a.data, cnst, b.data, a.length)
}

// a[i] += b[i] * c[i]
func MAdd2(a, b, c *Tensor) {
	AssertEqual(a.size, b.size)
	AssertEqual(b.size, c.size)
	dev.Madd2(a.data, b.data, c.data, a.length)
}

//func AddLinAnis(h, m *Tensor, K []*Tensor) {
//	dev.addLinAnis(h.comp[X].data, h.comp[Y].data, h.comp[Z].data,
//		m.comp[X].data, m.comp[Y].data, m.comp[Z].data,
//		K[XX].data, K[YY].data, K[ZZ].data,
//		K[YZ].data, K[XZ].data, K[XY].data,
//		h.length)
//}

// a[i]  = weightA * a[i] + weightB * b[i]
func LinearCombination(a, b *Tensor, weightA, weightB float32) {
	AssertEqual(a.size, b.size)
	dev.LinearCombination(a.data, b.data, weightA, weightB, a.length)
}

// a[i] += cnst
func AddConstant(a *Tensor, cnst float32) {
	dev.AddConstant(a.data, cnst, a.length)
}

//func AddLocalFields(m, h *Tensor, hext []float32, anisType int, anisK, anisAxes []float32) {
//	Assert(m.length == h.length)
//	Assert(len(m.size) == 4)
//	//fmt.Printf("hext:%v, anistype:%v, anisK:%v, anisAxes:%v\n", hext, anisType, anisK, anisAxes)
//	device.addLocalFields(m.data, h.data, hext, anisType, anisK, anisAxes, m.length/3)
//}


//func SemianalStep(min, mout, h *Tensor, dt, alpha float32) {
//	Assert(min.length == h.length)
//	Assert(min.length == mout.length)
//	device.semianalStep(min.data, mout.data, h.data, dt, alpha, min.length/3)
//}

//func OverrideStride(stride int) {
//	panic("OverrideStride is currently not compatible with the used FFT, it should always be 1")
//	Debugvv("Backend.OverrideStride(", stride, ")")
//	Assert(stride > 0 || stride == -1)
//	b.overrideStride(stride)
//}

// unsafe FFT
//func fftForward(plan uintptr, in, out uintptr) {
//	device.fft(plan, in, out, FFT_FORWARD)
//}
//
//// unsafe FFT
//func fftInverse(plan uintptr, in, out uintptr) {
//	device.fft(plan, in, out, FFT_INVERSE)
//}


// Takes an array size and returns the smallest multiple of Stride() where the array size fits in
//func (b  PadToStride(nFloats int) int {
//	stride := b.Stride()
//	gpulen := ((nFloats-1)/stride + 1) * stride
//
//	Assert(gpulen%stride == 0)
//	Assert(gpulen > 0)
//	Assert(gpulen >= nFloats)
//	return gpulen
//}
