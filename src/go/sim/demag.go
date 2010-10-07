//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import (
	"tensor"
	. "math"
)


/// converts from the full "rank-5" kernel format to the symmetric "array-of-rank3-tensors" format
// func FaceKernel6(unpaddedsize []int, cellsize []float, accuracy int) []*tensor.Tensor3 {
// 	k9 := FaceKernel(unpaddedsize, cellsize, accuracy)
// 	return toSymmetric(k9)
// }

func FaceKernel6(size []int, cellsize []float, accuracy int) []*tensor.Tensor3 {
	k := make([]*tensor.Tensor3, 6)
	for i := range k {
		k[i] = tensor.NewTensor3(size)
	}
	B := tensor.NewVector()
	R := tensor.NewVector()

	x1 := -(size[X] - 1) / 2
	x2 := size[X]/2 - 1
	// support for 2D simulations (thickness 1)
	if size[X] == 1 {
		x2 = 0
	}

	for s := 0; s < 3; s++ { // source index Ksdxyz
		for x := x1; x <= x2; x++ { // in each dimension, go from -(size-1)/2 to size/2 -1, wrapped. It's crucial that the unused rows remain zero, otherwise the FFT'ed kernel is not purely real anymore.
			xw := wrap(x, size[X])
			for y := -(size[Y] - 1) / 2; y <= size[Y]/2-1; y++ {
				yw := wrap(y, size[Y])
				for z := -(size[Z] - 1) / 2; z <= size[Z]/2-1; z++ {
					zw := wrap(z, size[Z])
					R.Set(float(x)*cellsize[X], float(y)*cellsize[Y], float(z)*cellsize[Z])

					faceIntegral(B, R, cellsize, s, accuracy)

					for d := s; d < 3; d++ { // destination index Ksdxyz
						i := KernIdx[s][d] // 3x3 symmetric index to 1x6 index
						k[i].Array()[xw][yw][zw] = B.Component[d]
					}
				}
			}
		}
	}
	return k
}

/// Integrates the demag field based on multiple points per face.
// func FaceKernel(size []int, cellsize []float, accuracy int) *tensor.Tensor5 {
// 	//size := PadSize(unpaddedsize)
// 	k := tensor.NewTensor5([]int{3, 3, size[0], size[1], size[2]})
// 	B := tensor.NewVector()
// 	R := tensor.NewVector()
// 
// 	for s := 0; s < 3; s++ { // source index Ksdxyz
// 		for x := -(size[X] - 1) / 2; x <= size[X]/2-1; x++ { // in each dimension, go from -(size-1)/2 to size/2 -1, wrapped. It's crucial that the unused rows remain zero, otherwise the FFT'ed kernel is not purely real anymore.
// 			xw := wrap(x, size[X])
// 			for y := -(size[Y] - 1) / 2; y <= size[Y]/2-1; y++ {
// 				yw := wrap(y, size[Y])
// 				for z := -(size[Z] - 1) / 2; z <= size[Z]/2-1; z++ {
// 					zw := wrap(z, size[Z])
// 					R.Set(float(x)*cellsize[X], float(y)*cellsize[Y], float(z)*cellsize[Z])
// 
// 					faceIntegral(B, R, cellsize, s, accuracy)
// 
// 					for d := 0; d < 3; d++ { // destination index Ksdxyz
// 						k.Array()[s][d][xw][yw][zw] = B.Component[d]
// 					}
// 
// 					//    if(xw == size[X]/2 + 1 || yw == size[Y]/2 + 1 || zw == size[Z]/2 + 1){
// 					//
// 					//       }
// 
// 				}
// 			}
// 		}
// 	}
// 	return k
// }


/**
 * Magnetostatic field at position r (integer, number of cellsizes away form source) for a given source magnetization direction m (X, Y, or Z)
 */
func faceIntegral(B, R *tensor.Vector, cellsize []float, s int, accuracy int) {
	n := accuracy                  // number of integration points = n^2
	u, v, w := s, (s+1)%3, (s+2)%3 // u = direction of source (s), v & w are the orthogonal directions
	R2 := tensor.NewVector()
	pole := tensor.NewVector() // position of point charge on the surface


	surface := cellsize[v] * cellsize[w] // the two directions perpendicular to direction s
	charge := surface

	pu1 := cellsize[u] / 2. // positive pole
	pu2 := -pu1             // negative pole

	B.Set(0., 0., 0.) // accumulates magnetic field
	for i := 0; i < n; i++ {
		pv := -(cellsize[v] / 2.) + cellsize[v]/float(2*n) + float(i)*(cellsize[v]/float(n))
		for j := 0; j < n; j++ {
			pw := -(cellsize[w] / 2.) + cellsize[w]/float(2*n) + float(j)*(cellsize[w]/float(n))

			pole.Component[u] = pu1
			pole.Component[v] = pv
			pole.Component[w] = pw

			R2.SetTo(R)
			R2.Sub(pole)
			r := R2.Norm()
			R2.Normalize()
			R2.Scale(charge / (4 * Pi * r * r))
			B.Add(R2)

			pole.Component[u] = pu2

			R2.SetTo(R)
			R2.Sub(pole)
			r = R2.Norm()
			R2.Normalize()
			R2.Scale(-charge / (4 * Pi * r * r))
			B.Add(R2)
		}
	}
	B.Scale(1. / (float(n * n))) // n^2 integration points
}
