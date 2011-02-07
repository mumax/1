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
	. "mumax/common"
	"fmt"
	"os"
	"tensor"
)


// Returns the kernel from cache if possible.
// Otherwise, a freshly calculated one is cached and returned.
func (s *Sim) LookupKernel(size []int, cellsize []float32, accuracy int, periodic []int) (kernel []*tensor.T3) {

	// If anything goes wrong unexpectedly, we just return a freshly calculated kernel.
	defer func() {
		err := recover()
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			kernel = FaceKernel6(size, cellsize, accuracy, periodic)
		}
	}()

	// empty widomdir means we must not use wisdom
	if s.wisdomdir == "" {
		kernel = FaceKernel6(size, cellsize, accuracy, periodic)
		return
	}

	// try to load cached kernel
	kerndir := s.wisdomdir + "/" + wisdomFileName(size, cellsize, accuracy, periodic)
	if fileExists(kerndir) {
		fmt.Println("using wisdom: ", kerndir)
		kernel = make([]*tensor.T3, 6)
		for i := range kernel {
			if s.needKernComp(i) {
				kernel[i] = s.loadKernComp(kerndir, i)
			}
		}
		fmt.Println("wisdom loaded")
		return
	} else {
		kernel = FaceKernel6(size, cellsize, accuracy, periodic)
		s.storeKernel(kernel, kerndir)
	}

	return
}


// INTERNAL: Loads kerndir/k**.tensor
func (s *Sim) loadKernComp(kerndir string, component int) *tensor.T3 {
	file := kerndir + "/k" + KernString[component] + ".tensor"
	in, err := os.Open(file, os.O_RDONLY, 0666)
	defer in.Close()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		panic(err)
	}
	return tensor.ToT3(tensor.Read(in))
}


// INTERNAL: Stores the kernel in kerndir
func (s *Sim) storeKernel(kernel []*tensor.T3, kerndir string) {
	// empty widomdir means we must not use wisdom
	if s.wisdomdir == "" {
		return
	}

	fmt.Println("storing wisdom: ", kerndir)

	// If anything goes wrong unexpectedly, then the kernel could not be saved,
	// but we should continue to run.
	defer func() {
		err := recover()
		if err != nil {
			fmt.Fprintln(os.Stderr, "Could not store kernel wisdom: ", err)
		}
	}()

	err1 := os.MkdirAll(kerndir, 0777)
	if err1 != nil {
		fmt.Fprintln(os.Stderr, err1)
	}

	for i := range kernel {
		if s.needKernComp(i) {

			file := kerndir + "/k" + KernString[i] + ".tensor"
			out, err := os.Open(file, os.O_CREATE|os.O_WRONLY, 0666)
			defer out.Close()
			if err != nil {
				fmt.Fprintln(os.Stderr, err)
				return
			}
			tensor.WriteBinary(out, kernel[i])
		}
	}
	fmt.Println("storing wisdom OK")
}


// INTERNAL returns a directory name (w/o absolute path) to store the kernel with given parameters
func wisdomFileName(size []int, cellsize []float32, accuracy int, periodic []int) string {
	pbc := ""
	if !(periodic[X] == 0 && periodic[Y] == 0 && periodic[Z] == 0) {
		pbc = fmt.Sprint("pbc", periodic[Z], "x", periodic[Y], "x", periodic[X])
	}
	return fmt.Sprint(size[Z], "x", size[Y], "x", size[X], "/",
		cellsize[Z], "x", cellsize[Y], "x", cellsize[X], "lex3",
		pbc,
		"acc", accuracy)
}

// INTERNAL
// Absolute path of the default kernel root directory:
// working directory + /kernelwisdom
func defaultWisdomDir() string {
	wd, err := os.Getwd()
	if err != nil {
		fmt.Fprintln(os.Stdout, err)
		return "" // don't use wisdom
	}
	return wd + "/wisdom"
}
