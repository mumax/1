//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import (
	"testing"
	"tensor"
	"fmt"
	"os"
)

func TestConv(t *testing.T) {

	size4D := []int{3, 1, 8, 8}
	size := size4D[1:]
	kernelSize := padSize(size)

	kernel := FaceKernel6(kernelSize, []float{1., 1., 1.}, 8)
	conv := NewConv(backend, size, kernel)

	for i := range conv.kernel {
		fmt.Println("K", i)
		if conv.kernel[i] != nil {
			tensor.Format(os.Stdout, conv.kernel[i])
		}
	}

	m, h := NewTensor(backend, size4D), NewTensor(backend, size4D)

	m.Set([]int{0, 0, 7, 7}, 1.)
	tensor.WriteFile("m.t", m)
	conv.Convolve(m, h)
	tensor.WriteFile("h.t", h)
}
