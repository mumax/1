//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim


// Kernel Wisdom:
// inspired by FFTW's "wisdom", which caches the optimal FFT plan parameters on disk,
// we store the micromagnetic kernels on disk.
//
// Whenever a kernel is needed, we first check whether it is already available on disk.
// If not, the kernel gets calculated and stored so that it can be re-used later.
//
// We store the untransformed demag kernel, so that we are compatible with different
// FFT layouts and exchange formulations.
//
import (
	"fmt"
	"os"
)

func (s *Sim) lookupKernel(size []int, cellsize []float32, accuracy int, periodic []int) {

}

// cellsize is stored as integer: picometer
const SIZEUNIT = 1e-12


// returns a directory name (w/o absolute path) to store the kernel with given parameters
func wisdomFileName(size []int, cellsize []float32, accuracy int, periodic []int) string {
	pbc := ""
	if !(periodic[X] == 0 && periodic[Y] == 0 && periodic[Z] == 0) {
		pbc = fmt.Sprint("pbc", periodic[Z], "x", periodic[Y], "x", periodic[X])
	}
	return fmt.Sprint(size[Z], "x", size[Y], "x", size[X], "/",
		int(cellsize[Z]/SIZEUNIT), "x", int(cellsize[Y]/SIZEUNIT), "x", int(cellsize[X]/SIZEUNIT), "pm3", "/",
		pbc,
		"accuracy", accuracy)
}

// Absolute path of the default kernel root directory:
// working directory + /kernelwisdom
func defaultWisdomDir() string {
	wd, err := os.Getwd()
	if err != nil {
		fmt.Fprintln(os.Stdout, err)
		return "" // don't use wisdom
	}
	return wd + "/kernelwisdom"

}
