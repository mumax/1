//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import (
	"tensor"
	"fmt"
)

const (
	X = 0
	Y = 1
	Z = 2
)

// A tensor on the calculating device (CPU, GPU),
// not directly accessible as a go array.
type DevTensor struct {
	*Backend ///< wraps the Device where the Tensor resides on (GPU/CPU/...)
	size     []int
	data     uintptr // points to float array on the GPU/CPU
}

// Allocates a new tensor on the device represented by Backend
func NewTensor(b *Backend, size []int) *DevTensor {
	t := new(DevTensor)
	t.Backend = b
	t.size = make([]int, len(size))
	length := 1
	for i := range size {
		t.size[i] = size[i]
		length *= size[i]
	}
	t.data = b.newArray(length)
	ZeroTensor(t)
	// TODO: runtime.SetFinalizer(t, Free)
	return t
}

// Wraps a pre-allocated device array in a tensor
func AsTensor(b *Backend, data uintptr, size []int) *DevTensor {
	return &DevTensor{b, size, data}
}

func (t *DevTensor) Get(index []int) float {
	i := tensor.Index(t.size, index)
	return t.arrayGet(t.data, i)
}


func (t *DevTensor) Set(index []int, value float) {
	i := tensor.Index(t.size, index)
	t.arraySet(t.data, i, value)
}

func (t *DevTensor) Size() []int {
	return t.size
}


func (t *DevTensor) Component(comp int) *DevTensor {
	assert(comp >= 0 && comp < t.size[0])
	size := t.size[1:]
	data := t.arrayOffset(t.data, comp*Len(size))
	return &DevTensor{t.Backend, size, data}
}


func Len(size []int) int {
	prod := 1
	for _, s := range size {
		prod *= s
	}
	return prod
}


func assertEqualSize(sizeA, sizeB []int) {
	assert(len(sizeA) == len(sizeB))
	for i := range sizeA {
		assert(sizeA[i] == sizeB[i])
	}
}


/// copies between two Tensors on the sim
func TensorCopyOn(source, dest *DevTensor) {
	assert(tensor.EqualSize(source.size, dest.size))
	source.memcpyOn(source.data, dest.data, tensor.N(source))
}

/// copies a tensor to the GPU
func TensorCopyTo(source tensor.StoredTensor, dest *DevTensor) {
	///@todo sim.Set(), allow tensor.Tensor source, type switch for efficient copying
	///@todo TensorCpy() with type switch for auto On/To/From
	assert(tensor.EqualSize(source.Size(), dest.size))
	dest.memcpyTo(&(source.List()[0]), dest.data, tensor.N(source))
}

/// copies a tensor to the GPU
func TensorCopyFrom(source *DevTensor, dest tensor.StoredTensor) {
	///@todo sim.Set(), allow tensor.Tensor source, type switch for efficient copying
	///@todo TensorCpy() with type switch for auto On/To/From
	assert(tensor.EqualSize(source.Size(), dest.Size()))
	source.memcpyFrom(source.data, &(dest.List()[0]), tensor.N(source))
}


func ZeroTensor(t *DevTensor) {
	t.zero(t.data, tensor.N(t))
}


func CopyPad(source, dest *DevTensor) {
	source.copyPad(source.data, dest.data, source.size, dest.size)
}


func CopyUnpad(source, dest *DevTensor) {
	source.copyUnpad(source.data, dest.data, source.size, dest.size)
}


func (t *DevTensor) String() string {
	return fmt.Sprint("Tensor{", t.size, "}")
}
