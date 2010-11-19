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
	"tensor"
)

func (s *Sim) lookupKernel(size []int, cellsize []float32, accuracy int, periodic []int) (kernel []*tensor.T3) {
  // If anything goes wrong unexpectedly, we just return a freshly calculated kernel.
  defer func() {
		err := recover()
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			kernel = FaceKernel6(size, cellsize, accuracy, periodic)
		}
	}()
	

	kerndir := s.wisdomdir + "/" + wisdomFileName(size, cellsize, accuracy, periodic)
	fmt.Println("using wisdom: ", kerndir)

	if fileExists(kerndir) {
		kernel = make([]*tensor.T3, 6)
		for i := range kernel {
			if s.needKernComp(i) {

			}
		}
	} else {
		kernel = FaceKernel6(size, cellsize, accuracy, periodic)
		storeKernel(kernel, kerndir)
	}

	return
}

func storeKernel(kernel []*tensor.T3, kerndir string){
  // If anything goes wrong unexpectedly, then the kernel could not be saved,
  // but we should continue to run.
  defer func(){
    err := recover()
    if err != nil{
      fmt.Fprintln(os.Stderr, "Could not store kernel wisdom: ", err)
    }
  }()

  
  
}

// returns a directory name (w/o absolute path) to store the kernel with given parameters
func wisdomFileName(size []int, cellsize []float32, accuracy int, periodic []int) string {
	pbc := ""
	if !(periodic[X] == 0 && periodic[Y] == 0 && periodic[Z] == 0) {
		pbc = fmt.Sprint("pbc", periodic[Z], "x", periodic[Y], "x", periodic[X])
	}
	return fmt.Sprint(size[Z], "x", size[Y], "x", size[X], "/",
		cellsize[Z], "x", cellsize[Y], "x", cellsize[X], "lex3", "/",
		pbc,
		"acc", accuracy)
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
