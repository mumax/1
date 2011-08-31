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
	"mumax/tensor"
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


// input is assumed vector field
func subsample4(data *tensor.T4, small *tensor.T4, f int) {
	bigsize := data.Size()
	smallsize := []int{3, bigsize[1] / f, bigsize[2] / f, bigsize[3] / f}
	for i := range smallsize {
		if smallsize[i] < 1 {
			smallsize[i] = 1
		}
	}
	A := data.Array()  // big array
	a := small.Array() // small array

	// reset small array before adding to it
	sl := small.List()
	for i := range sl {
		sl[i] = 0
	}

	for c := range a {

		for i := range a[c] {
			for j := range a[c][i] {
				for k := range a[c][i][j] {

					n := 0

					for I := i * f; I < min((i+1)*f, bigsize[1]); I++ {
						for J := j * f; J < min((j+1)*f, bigsize[2]); J++ {
							for K := k * f; K < min((k+1)*f, bigsize[3]); K++ {
								n++
								a[c][i][j][k] += A[c][I][J][K]
							}
						}
					}
					a[c][i][j][k] /= float32(n)
				}
			}
		}
	}
}


func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}


func resample3(in *tensor.T3, size2 []int) *tensor.T3 {
	size1 := in.Size()
	assert(len(size2) == 3)
	assert(len(size1) == 3)

	out := tensor.NewT3(size2)

	out_a := out.Array()
	in_a := in.Array()

	for i := range out_a {
		i1 := (i * size1[0]) / size2[0]
		for j := range out_a[i] {
			j1 := (j * size1[1]) / size2[1]
			for k := range out_a[i][j] {
				k1 := (k * size1[2]) / size2[2]
				out_a[i][j][k] = in_a[i1][j1][k1]
			}
		}
	}

	return out
}
