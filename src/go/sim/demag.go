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

// Calculates the magnetostatic kernel
//
// size: size of the kernel, usually 2 x larger than the size of the magnetization due to zero padding
// accuracy: use 2^accuracy integration points
//
// return value: A symmetric rank 5 tensor K[sourcedir][destdir][x][y][z]
// (e.g. K[X][Y][1][2][3] gives H_y at position (1, 2, 3) due to a unit dipole m_x at the origin.
// Only the non-redundant elements of this symmetric tensor are returned: XX, YY, ZZ, YZ, XZ, XY
// You can use the function KernIdx to convert from source-dest pairs like XX to 1D indices:
// K[KernIdx[X][X]] returns K[XX]
func FaceKernel6(size []int, cellsize []float32, accuracy int) []*tensor.T3 {
	k := make([]*tensor.T3, 6)
	for i := range k {
		k[i] = tensor.NewT3(size)
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
					R.Set(float32(x)*cellsize[X], float32(y)*cellsize[Y], float32(z)*cellsize[Z])

					faceIntegral(B, R, cellsize, s, accuracy)

					for d := s; d < 3; d++ { // destination index Ksdxyz
						i := KernIdx[s][d] // 3x3 symmetric index to 1x6 index
						k[i].Array()[xw][yw][zw] = B.Component[d]
					}
				}
			}
		}
	}
	for s:=0; s<3; s++{
    for d:=0; d<3; d++{
      assert(k[KernIdx[s][d]].Array()[0][0][0] == selfKernel(s, cellsize, accuracy)[d])
    }}
	return k
}

// Calculates only the self-kernel K[ij][0][0][0].
// used for edge corrections where we need to subtract this generic self kernel contribution and
// replace it by an edge-corrected version specific for each cell.
func selfKernel(sourcedir int, cellsize []float32, accuracy int) []float32{
  B := tensor.NewVector()
  R := tensor.NewVector()
  faceIntegral(B, R, cellsize, sourcedir, accuracy)
  return []float32{B.Component[X], B.Component[Y], B.Component[Z]}
}


// Magnetostatic field at position r (integer, number of cellsizes away form source) for a given source magnetization direction m (X, Y, or
// s = source direction (x, y, z)
func faceIntegral(B, R *tensor.Vector, cellsize []float32, s int, accuracy int) {
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
		pv := -(cellsize[v] / 2.) + cellsize[v]/float32(2*n) + float32(i)*(cellsize[v]/float32(n))
		for j := 0; j < n; j++ {
			pw := -(cellsize[w] / 2.) + cellsize[w]/float32(2*n) + float32(j)*(cellsize[w]/float32(n))

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
	B.Scale(1. / (float32(n * n))) // n^2 integration points
}
