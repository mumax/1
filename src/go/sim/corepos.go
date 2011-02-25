//  This file is part of MuMax, a high-performance micromagnetic simulator
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

// This file implements finding the vortex core position

import (
	//. "mumax/common"
	//"mumax/tensor"
)

func (s *Sim) corePos() (pos [2]float32) {
	mz := s.mLocal.TArray[0][0]
	var max float32 = -1.
	maxX, maxY := 0, 0
	for y := range mz{
		for x := range mz[y]	{
			m := mz[y][x]
			if m > max{
				maxX, maxY = x, y
				max = m
			}
		}
	}
	pos[0] = float32(maxX)+ interpolate_maxpos(max, -1., mz[maxY][maxX-1], 1., mz[maxY][maxX+1])
	pos[1] = float32(maxY)+ interpolate_maxpos(max, -1., mz[maxY-1][maxX], 1., mz[maxY+1][maxX])

	return
}

func interpolate_maxpos(f0, d1, f1, d2, f2 float32) float32 {
	b := (f2 - f1) / (d2 - d1)
	a := ((f2-f0)/d2 - (f0-f1)/(-d1)) / (d2 - d1)
	return -b / (2 * a)
}
