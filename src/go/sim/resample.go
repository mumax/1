//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

// Functions for re-sampling a tensor in case
// its size does not match the desired size.
// E.g.: initial magnetization input file does
// not have the suited size.

import (
	"tensor"
)

// input is assumed vector field
func resample4(in *tensor.T4, size2 []int) *tensor.T4 {
	assert(len(size2) == 4)
	assert(size2[0] == 3)
	out := tensor.NewT4(size2)
	out_a := out.Array()
	in_a := in.Array()
	size1 := in.Size()
	for c := range out_a {
		for i := range out_a[c] {
			i1 := (i * size1[1]) / size2[1]
			for j := range out_a[0][i] {
				j1 := (j * size1[2]) / size2[2]
				for k := range out_a[0][i][j] {
					k1 := (k * size1[3]) / size2[3]
					out_a[c][i][j][k] = in_a[c][i1][j1][k1]
				}
			}
		}
	}
	return out
}


func resample3(in *tensor.T3, size2 []int) *tensor.T3 {
	assert(len(size2) == 3)
	assert(size2[0] == 3)
	out := tensor.NewT3(size2)
	out_a := out.Array()
	in_a := in.Array()
	size1 := in.Size()
	for c := range out_a {
		for i := range out_a[c] {
			i1 := (i * size1[1]) / size2[1]
			for j := range out_a[0][i] {
				j1 := (j * size1[2]) / size2[2]
				out_a[c][i][j] = in_a[c][i1][j1]
			}
		}
	}

	return out
}
