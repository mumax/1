//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package tensor

// The tensor interface specifies the size of the tensor
// and the data as a contiguous list.
type Interface interface {
	Size() []int     // The size of the tensor in each direction.
	List() []float32 // The underlying data. Its length must be Size()[0] * Size()[1] * ... * Size()[N]
}
