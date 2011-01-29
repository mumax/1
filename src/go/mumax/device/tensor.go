//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package device

import (
	. "mumax/common"
	"mumax/tensor"
	"fmt"
	"runtime"
	"unsafe"
)


// A rank 4 tensor on the device (CPU, GPU),
// not directly accessible as a go array.
// Scalar fields are represented by a 1 x N1 x N2 x N3 tensor
// Vector fields are represented by a 3 x N1 x N2 x N3 tensor
// Tensor fields are represented by a 6 x N1 x N2 x N3 tensor
type Tensor struct {
	size4  [4]int    // number of elements in each dimension
	size   []int     // size4 as slice
	length int       // total number of floats (product of elements of size) 
	data   uintptr   // points to float array on the device
	comp   []*Tensor // wraps the components (scalar fields). E.g. mx = m.comp[0].
}


// Allocates a new vector/tensor field on the device.
// E.g.: NewVector([]int{3, 32, 32, 32}) for a 3-vector field
// NewVector(6, ...) for a symmetric tensor field.
func NewTensor(size []int) *Tensor {
	Assert(len(size) == 4)
	t := new(Tensor)
	t.size = t.size4[:]
	copy(t.size, size) // dest, source
	t.length = Len(size)
	complen := Len(size[1:])
	comp_ptrs := device.NewArray(size[0], complen)

	t.data = comp_ptrs[0]
	t.Zero()

	//initialize component list
	t.comp = make([]*Tensor, size[0])
	compsize := []int{1, size[1], size[2], size[3]}
	for i := range t.comp {
		t.comp[i] = new(Tensor)
		t.comp[i].size = t.comp[i].size4[:]
		copy(t.comp[i].size, compsize) // dest, source

		t.comp[i].length = complen
		t.comp[i].data = comp_ptrs[i]
	}

	runtime.SetFinalizer(t, Free)
	return t
}


// Frees the underlying storage.
// It is safe to double-free.
func (t *Tensor) Free() {
	if t.data != 0 {
		device.FreeArray(t.data)
		t.data = 0
	}
}

// Free() method as function so it can be passed to	runtime.SetFinalizer
func Free(t *Tensor) {
	t.Free()
}


// Wraps a pre-allocated device array in a tensor
// comp remains uninitialized
//func AsTensor(b *Backend, data uintptr, size []int) *Tensor {
//	return &Tensor{b, size, Len(size), data, nil}
//}

// func (t *Tensor) Get(index []int) float32 {
// 	i := tensor.Index(t.size, index)
// 	return t.arrayGet(t.data, i)
// }


// func (t *Tensor) Set(index []int, value float32) {
// 	i := tensor.Index(t.size, index)
// 	t.arraySet(t.data, i, value)
// }

func (t *Tensor) Size() []int {
	return t.size
}


// func (t *Tensor) Component(comp int) *Tensor {
// 	Assert(comp >= 0 && comp < t.size[0])
// 	size := t.size[1:]
// 	data := t.arrayOffset(t.data, comp*Len(size))
// 	return &Tensor{t.Backend, size, data}
// }


func Len(size []int) int {
	prod := 1
	for _, s := range size {
		prod *= s
	}
	return prod
}


func AssertEqualSize(sizeA, sizeB []int) {
	Assert(len(sizeA) == len(sizeB))
	for i := range sizeA {
		Assert(sizeA[i] == sizeB[i])
	}
}


// copies between two Tensors on the device
func CopyOn(source, dest *Tensor) {
	AssertEqual(source.size, dest.size)
	device.Memcpy(source.data, dest.data, source.length, CPY_ON)
}

// copies a tensor to the device
// TODO Copy() with type switch for auto On/To/From
func CopyTo(source tensor.Interface, dest *Tensor) {
	AssertEqual(source.Size(), dest.size)
	device.Memcpy(uintptr(unsafe.Pointer(&(source.List()[0]))), dest.data, dest.length, CPY_TO)
}

// copies a tensor from the device
func CopyFrom(source *Tensor, dest tensor.Interface) {
	AssertEqual(source.Size(), dest.Size())
	device.Memcpy(source.data, uintptr(unsafe.Pointer(&(dest.List()[0]))), source.length, CPY_FROM)
}


// Sets all elements to zero
func (t *Tensor) Zero() {
	device.Zero(t.data, t.length)
}


//func CopyPad(source, dest *Tensor) {
//	device.copyPad(source.data, dest.data, source.size, dest.size)
//}
//
//
//func CopyUnpad(source, dest *Tensor) {
//	device.copyUnpad(source.data, dest.data, source.size, dest.size)
//}


func (t *Tensor) String() string {
	return fmt.Sprintf("%#v", t)
}

// m[X] instead of m[0] is more clear
const (
	X = 0
	Y = 1
	Z = 2
)

// Maps the 3x3 indices of the symmetric demag kernel (K_ij) onto
// a length 6 array containing the upper triangular part:
// (Kxx, Kyy, Kzz, Kyz, Kxz, Kxy)
const (
	XX = 0
	YY = 1
	ZZ = 2
	YZ = 3
	XZ = 4
	XY = 5
)
