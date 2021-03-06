//  This file is part of MuMax, a high-performance micromagnetic simulator
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

// This file implements finding the vortex core position

import (
	. "mumax/common"
	//"mumax/tensor"
)


// Returns the vortex core position of the sim's mLocal.
// x,y coordinate (user axes) expressed in meters, 0,0 is center of grid.
func (s *Sim) corePos() (pos [2]float32) {
	mz := s.mLocal.TArray[0][0]
	var max float32 = -1.
	maxX, maxY := 0, 0
	for y := 1; y < len(mz)-1; y++ { // Avoid the boundaries so the neighbor interpolation can't go out of bounds.
		for x := 1; x < len(mz[y])-1; x++ {
			m := Abs(mz[y][x])
			if m > max {
				maxX, maxY = x, y
				max = m
			}
		}
	}
	pos[0] = float32(maxX) + interpolate_maxpos(max, -1., Abs(mz[maxY][maxX-1]), 1., Abs(mz[maxY][maxX+1])) - float32(len(mz[1]))/2.
	pos[1] = float32(maxY) + interpolate_maxpos(max, -1., Abs(mz[maxY-1][maxX]), 1., Abs(mz[maxY+1][maxX])) - float32(len(mz[0]))/2.

	pos[0] *= s.input.cellSize[2]
	pos[1] *= s.input.cellSize[1]
	return
}

func interpolate_maxpos(f0, d1, f1, d2, f2 float32) float32 {
	b := (f2 - f1) / (d2 - d1)
	a := ((f2-f0)/d2 - (f0-f1)/(-d1)) / (d2 - d1)
	return -b / (2 * a)
}
