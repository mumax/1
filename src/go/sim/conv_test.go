//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import (
	"testing"
	"tensor"
	"os"
)

func TestConv(t *testing.T) {

	size4D := []int{3, 2, 8, 8}
	size := size4D[1:]
	kernelSize := padSize(size)

	kernel := ZeroKernel6(kernelSize)
	kernel[XX].Array()[0][0][0] = 1.
	kernel[XX].Array()[0][0][1] = 1.

	kernel[YY].Array()[0][0][0] = 0.
	kernel[ZZ].Array()[0][0][0] = 0.

	conv := NewConv(backend, size, kernel)

	mLocal := tensor.NewT4(size4D)
	mLocal.Array()[X][0][0][0] = 1.
	mLocal.Array()[Y][0][0][0] = 0.
	mLocal.Array()[Z][0][0][0] = 3.

	mLocal.WriteTo(os.Stdout)

	m, h := NewTensor(backend, size4D), NewTensor(backend, size4D)
	TensorCopyTo(mLocal, m)

	conv.Convolve(m, h)

	hLocal := tensor.NewT4(size4D)
	TensorCopyFrom(h, hLocal)

	hLocal.WriteTo(os.Stdout)
}
