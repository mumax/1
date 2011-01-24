//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.


package main

import (
	"omf"
	"iotool"
	"fmt"
)

var coreodt *omf.TabWriter

// Finds vortex core position in file.
// pol = 1 : up
// pol = -1: down
func CorePos(fname string, pol float32) {
	init_odt_corepos(fname)

	mz := data.TArray[Z]

	// find coarse maximum (or minimum)
	max := pol * mz[0][0][0]
	maxX, maxY, maxZ := 0, 0, 0
	for i := range mz {
		for j := 1; j < len(mz[i])-1; j++ {
			for k := 1; k < len(mz[i][j])-1; k++ {
				if pol*mz[i][j][k] > max {
					max = pol * mz[i][j][k]
					maxX, maxY, maxZ = i, j, k
				}
			}
		}
	}
	fmt.Println(maxZ)
	// then interpolate around the top
	corex := float32(maxX) //+ interpolate_maxpos(max, -1., pol*mz[maxZ][maxX-1][maxY], 1., pol*mz[maxZ][maxX+1][maxY])
	corey := float32(maxY) //+ interpolate_maxpos(max, -1., pol*mz[maxZ][maxX][maxY-1], 1., pol*mz[maxZ][maxX][maxY+1])

	// and express in length units
	//	cellsizex := info.StepSize[0]
	//	cellsizey := info.StepSize[0]
	//	cellsizex /= float32(len(mz[0]))
	//	cellsizey /= float32(len(mz[0][0]))
	//	corex *= float32(cellsizex)
	//	corey *= float32(cellsizey)

	// oops, turns out we were transposed all the time
	//corex, corey = corey, corex

	time := info.DescGetFloat32("time")
	coreodt.Print(time)
	coreodt.Print(corex)
	coreodt.Print(corey)
	coreodt.Flush()
	//coreodt.Print(id)

}

func init_odt_corepos(filename string) {
	if coreodt == nil {
		out := iotool.MustOpenWRONLY(filename)
		coreodt = omf.NewTabWriter(out)
		coreodt.AddColumn("Time", "s")
		coreodt.AddColumn("CoreX", "m")
		coreodt.AddColumn("CoreY", "m")
		//	odt.AddColumn("id", "")
	}
}

func interpolate_maxpos(f0, d1, f1, d2, f2 float32) float32 {
	b := (f2 - f1) / (d2 - d1)
	a := ((f2-f0)/d2 - (f0-f1)/(-d1)) / (d2 - d1)
	return -b / (2 * a)
}
