package sim

/**
 * A kernel is a rank 5 Tensor: K[S][D][x][y][z].
 * S and D are source and destination directions, ranging from 0 (X) to 2 (Z).
 * K[S][D][x][y][z] is the D-the component of the magnetic field at position
 * (x,y,z) due to a unit spin along direction S, at the origin.
 *
 * The kernel is twice the size of the magnetization field we want to convolve it with.
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
func UnitKernel(unpaddedsize []int) tensor.StoredTensor {
	size := PadSize(unpaddedsize)
	k := tensor.NewTensor5([]int{3, 3, size[0], size[1], size[2]})
	for c := 0; c < 3; c++ {
		k.Array()[c][c][0][0][0] = 1.
	}
	return k
}


/* --------- Internal functions --------- */

func toSymmetric(k9 tensor.StoredTensor) []tensor.StoredTensor {
	k9X := tensor.Component(k9, X)
	k9Y := tensor.Component(k9, Y)
	k9Z := tensor.Component(k9, Z)

	k6 := make([]tensor.StoredTensor, 6)

	k6[XX] = tensor.Buffer(tensor.Component(k9X, X))
	k6[YY] = tensor.Buffer(tensor.Component(k9Y, Y))
	k6[ZZ] = tensor.Buffer(tensor.Component(k9Z, Z))
	k6[YZ] = tensor.Buffer(tensor.Component(k9Y, Z))
	k6[XZ] = tensor.Buffer(tensor.Component(k9X, Z))
	k6[XY] = tensor.Buffer(tensor.Component(k9X, Y))

	return k6
}

func wrap(number, max int) int {
	for number < 0 {
		number += max
	}
	return number
}


func FSqrt(x float64) float {
	return float(Sqrt(x))
}

func PadSize(size []int) []int {
	paddedsize := make([]int, len(size))
	for i := range size {
		// THIS SEEMS TO BREAK THINGS:
		//     if(size[i] < 4){
		//       paddedsize[i] = 2*size[i]-1; // sizes 1, 2, 3, 4, 5 should be OK
		//     }
		//      else{
		paddedsize[i] = 2 * size[i]
		//}
	}
	return paddedsize
}
