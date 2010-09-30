//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package tensor

type Resampled struct {
	original Tensor
	size     []int
}

func (t *Resampled) Size() []int {
	return t.size
}

func (t *Resampled) Get(index []int) float {
	index2 := make([]int, len(index))
	for i := range index {
		index2[i] = (index[i] * t.size[i]) / t.original.Size()[i]
	}
	return t.original.Get(index2)
}

func Resample(t Tensor, size []int) Tensor {
	return &Resampled{t, size}
}
