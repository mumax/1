//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package tensor

import (
	. "math"
)

//_____________________________________________________________________ slice

/** Slice a tensor by fixing a dimension to value.
 * @note Can be used to take a component.
 */
func Slice(t Tensor, dim, value int) *TensorSlice {
	return &TensorSlice{t, dim, value, nil}
}

func Component(t Tensor, component int) *TensorSlice {
	return Slice(t, 0, component)
}

type TensorSlice struct {
	Tensor         // the original tensor
	dim, value int // the dimension to slice away by fixing it to a value
	size       []int
}

func (t *TensorSlice) Size() []int {
	if t.size == nil {
		origsize := t.Tensor.Size()
		size := make([]int, Rank(t.Tensor)-1)
		for i := 0; i < t.dim; i++ {
			size[i] = origsize[i]
		}
		for i := t.dim + 1; i < len(origsize); i++ {
			size[i-1] = origsize[i]
		}
		t.size = size
	}
	return t.size
}

func (t *TensorSlice) Get(index []int) float {

	bigindex := make([]int, Rank(t.Tensor))
	for i := 0; i < t.dim; i++ {
		bigindex[i] = index[i]
	}
	bigindex[t.dim] = t.value
	for i := t.dim + 1; i < len(bigindex); i++ {
		bigindex[i] = index[i-1]
	}
	return t.Tensor.Get(bigindex)
}

//_____________________________________________________________________ transpose

/** Swap two dimensions. */

func Transpose(t Tensor, x, y int) *TransposedTensor {
	return &TransposedTensor{t, x, y, nil}
}

type TransposedTensor struct {
	Tensor
	x, y int
	size []int
}

func (t *TransposedTensor) Get(index []int) (v float) {
	// swap
	index[t.x], index[t.y] = index[t.y], index[t.x]
	v = t.Tensor.Get(index)
	// swap back
	index[t.x], index[t.y] = index[t.y], index[t.x]
	return
}

func (t *TransposedTensor) Size() []int {
	if t.size == nil {
		origsize := t.Tensor.Size()
		size := make([]int, len(origsize))
		for i := range size {
			size[i] = origsize[i]
		}
		size[t.x], size[t.y] = size[t.y], size[t.x]
		t.size = size
	}
	return t.size
}

//_____________________________________________________________________ normalize

func Normalize(t Tensor, dim int) *NormalizedTensor {
	return &NormalizedTensor{t, dim}
}

type NormalizedTensor struct {
	Tensor
	dim int
}

func (t *NormalizedTensor) Get(index []int) float {
	size := t.Tensor.Size()

	// make an index for going through the direction over which we normalize
	index2 := make([]int, len(size))
	for i := range index2 {
		index2[i] = index[i]
	}

	// accumulate the total norm of all data along that direction
	var norm2 float64 = 0.
	for i := 0; i < size[t.dim]; i++ {
		index2[t.dim] = i
		value := t.Tensor.Get(index2)
		norm2 += float64(value * value)
	}

	return t.Tensor.Get(index) / float(Sqrt(norm2))
}

//_____________________________________________________________________ average

func Average(t Tensor, dim int) *TensorAverage {
	return &TensorAverage{t, dim, nil}
}

type TensorAverage struct {
	Tensor     // the original tensor
	dim    int // the dimension to average away
	size   []int
}

func (t *TensorAverage) Size() []int {
	if t.size == nil {
		origsize := t.Tensor.Size()
		size := make([]int, Rank(t.Tensor)-1)
		for i := 0; i < t.dim; i++ {
			size[i] = origsize[i]
		}
		for i := t.dim + 1; i < len(origsize); i++ {
			size[i-1] = origsize[i]
		}
		t.size = size
	}
	return t.size
}

func (t *TensorAverage) Get(index []int) float {

	bigindex := make([]int, Rank(t.Tensor))
	for i := 0; i < t.dim; i++ {
		bigindex[i] = index[i]
	}
	for i := t.dim + 1; i < len(bigindex); i++ {
		bigindex[i] = index[i-1]
	}

	var sum float64 = 0.
	for i := 0; i < t.Tensor.Size()[t.dim]; i++ {
		bigindex[t.dim] = i
		sum += float64(t.Tensor.Get(bigindex))
	}
	return float(sum / float64(t.Tensor.Size()[t.dim]))
}

//_____________________________________________________________________ sum

type TensorSum struct {
	t1, t2 Tensor
}

func (t *TensorSum) Size() []int { return t.t1.Size() }

func (t *TensorSum) Get(index []int) float { return t.t1.Get(index) + t.t2.Get(index) }

func Add(t1, t2 Tensor) *TensorSum {
	assert(EqualSize(t1.Size(), t2.Size()))
	return &TensorSum{t1, t2}
}


//_____________________________________________________________________ total

/** sum of all elements */

type TensorTotal struct {
	original Tensor
}

func (t *TensorTotal) Size() []int {
	return []int{} // a scalar
}

func (t *TensorTotal) Get(index []int) float {
	assert(len(index) == 0)

	sum := float64(0.0)

	for it := NewIterator(t); it.HasNext(); it.Next() {
		sum += float64(t.Get(it.Index()))
	}
	return float(sum)
}

func Total(t Tensor) *TensorTotal { return &TensorTotal{t} }
