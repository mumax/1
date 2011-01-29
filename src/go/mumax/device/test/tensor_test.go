//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package test

import (
	"testing"
	"mumax/tensor"
)




func TestCopy(test *testing.T){
	UseTestDevice()

	size := []int{3, 4, 32, 64}
	loc := tensor.NewT4(size)
	dev := NewTensor(size)
	CopyTo(loc, dev)
	CopyFrom(dev, loc)
}
