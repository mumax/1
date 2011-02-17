//  This file is part of MuMax, a high-performance micromagnetic simulator
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package main

// Calculates a micromagnetic kernel and dumps it to stdout,
// to be read by kernelpipe.go

import (
	. "mumax/common"
	"mumax/tensor"
	"sim"
	"flag"
	"os"
)


func main() {
	flag.Parse()
	if flag.NArg() != 10 {
		panic(InputErr("Need 10 command-line arguments"))
	}

	size := []int{0, 0, 0}
	cellSize := []float32{0, 0, 0}
	periodic := []int{0, 0, 0}

	size[X] = Atoi(flag.Arg(0))
	size[Y] = Atoi(flag.Arg(1))
	size[Z] = Atoi(flag.Arg(2))
	cellSize[X] = Atof32(flag.Arg(3))
	cellSize[Y] = Atof32(flag.Arg(4))
	cellSize[Z] = Atof32(flag.Arg(5))
	periodic[X] = Atoi(flag.Arg(6))
	periodic[Y] = Atoi(flag.Arg(7))
	periodic[Z] = Atoi(flag.Arg(8))
	// nThreads not used

	demag := sim.FaceKernel6(size, cellSize, 8, periodic)

	for i := range demag {
		tensor.Write(os.Stdout, demag[i])
	}
}
