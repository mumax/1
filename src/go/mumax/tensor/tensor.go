//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package tensor

// This file provides various implementations of tensor.Interface

import (
	"fmt"
)


// A tensor.T represents an arbitrary-rank tensor.
type T struct {
	TSize []int
	TList []float32
}


func NewT(size []int) *T {
	t := new(T)
	t.Init(size)
	return t
}

func (t *T) Init(size []int) {
	t.TSize = size
	t.TList = make([]float32, Prod(size))
}

// Converts a tensor.Interface to a tensor.T
func ToT(t Interface) *T {
	return &T{t.Size(), t.List()}
}

func (t *T) Size() []int {
	return t.TSize
}


func (t *T) List() []float32 {
	return t.TList
}


// A rank 4 tensor.
type T4 struct {
	T                      // The underlying arbitrary-rank tensor
	TArray [][][][]float32 // The data list represented as an array, for convenience.
}

func (t *T4) Init(size []int) {
	if len(size) != 4 {
		panic("Illegal argument: " + fmt.Sprint(size))
	}
	t.TSize = size
	t.TList, t.TArray = Array4D(size[0], size[1], size[2], size[3])
}

func NewT4(size []int) *T4 {
	t := new(T4)
	t.Init(size)
	return t
}

func (t *T4) Array() [][][][]float32 {
	return t.TArray
}

// Converts a tensor.Interface to a tensor.T
func ToT4(t Interface) *T4 {
	return &T4{*ToT(t), Slice4D(t.List(), t.Size())}
}


// A rank 3 tensor.
type T3 struct {
	T                    // The underlying arbitrary-rank tensor
	TArray [][][]float32 // The data list represented as an array, for convenience.
}

func (t *T3) Init(size []int) {
	if len(size) != 3 {
		panic("Illegal argument")
	}
	t.TSize = size
	t.TList, t.TArray = Array3D(size[0], size[1], size[2])
}

func NewT3(size []int) *T3 {
	t := new(T3)
	t.Init(size)
	return t
}

func (t *T3) Array() [][][]float32 {
	return t.TArray
}

func ToT3(t Interface) *T3 {
	return &T3{*ToT(t), Slice3D(t.List(), t.Size())}
}


// Returns size[0] * size[1] * ... * size[N]
func Prod(size []int) int {
	prod := 1
	for _, s := range size {
		prod *= s
	}
	return prod
}

// The total number of elements.
func Len(t Interface) int {
	return Prod(t.Size())
}

// Returns the rank of the tensor
// (number of indices).
func Rank(t Interface) int {
	return len(t.Size())
}
