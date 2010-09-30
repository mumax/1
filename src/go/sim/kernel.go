//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

// TODO
// kernel is a bit of a mess:
//  * we only need to calculate the upper triangular part
//  * we need to use the symmetry to make the calculation 8x faster
//  * return only 1/8 of the total kernel and have a function to obtain the rest via mirroring (good for storing them)
//  * return []Tensor3 right away without conversion

/**
 * A kernel is a rank 5 Tensor: K[S][D][x][y][z].
 * S and D are source and destination directions, ranging from 0 (X) to 2 (Z).
 * K[S][D][x][y][z] is the D-the component of the magnetic field at position
 * (x,y,z) due to a unit spin along direction S, at the origin.
 *
 * As the kernel is symmetric Ksd == Kds, we only work with the upper-triangular part
 *
 * The kernel is usually twice the size of the magnetization field we want to convolve it with.
 * The indices are wrapped: a negative index i is stored at N-abs(i), with N
 * the total size in that direction.
 *
 * Idea: we migth calculate in the kernel in double precession and only round it
 * just before it is returned, or even after doing the FFT. Because it is used over
 * and over, this small gain in accuracy *might* be worth it.
 */

import (
	"tensor"
	. "math"
)

/** Unit kernel, for debugging. */
func UnitKernel(paddedsize []int) tensor.StoredTensor {
	size := paddedsize
	k := tensor.NewTensor5([]int{3, 3, size[0], size[1], size[2]})
	for c := 0; c < 3; c++ {
		k.Array()[c][c][0][0][0] = 1.
	}
	return k
}

// Modulo-like function:
// Wraps an index to [0, max] by adding/subtracting a multiple of max.
func wrap(number, max int) int {
	//fmt.Print("wrap(", number, max, ")=")
	for number < 0 {
		number += max
	}
	for number >= max {
		number -= max
	}
	// 	fmt.Println(number)
	return number
}


func FSqrt(x float64) float {
	return float(Sqrt(x))
}

// Add padding x 2 in all directions, except when a dimension == 1 (no padding neccesary)
func padSize(size []int) []int {
	paddedsize := make([]int, len(size))
	for i := range size {
		if size[i] > 1 {
			paddedsize[i] = 2 * size[i]
		} else {
			paddedsize[i] = size[i]
		}
	}
	return paddedsize
}

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

// Maps the 3x3 indices of the symmetric demag kernel (K_ij) onto
// a length 6 array containing the upper triangular part:
// (Kxx, Kyy, Kzz, Kyz, Kxz, Kxy)
// Invalid (unused) indices like ZX return -1.
var KernIdx [3][3]int = [3][3]int{
	[3]int{XX, XY, XZ},
	[3]int{-1, YY, YZ},
	[3]int{-1, -1, ZZ}}
