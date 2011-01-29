//  This file is part of MuMax, a high-perfomrance micromagnetic simulator.
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package common

// Size, in bytes, of a C single-precision float
const SIZEOF_CFLOAT = 4

// Go equivalent of &array[index] (for a float array).
func ArrayOffset(array uintptr, index int) uintptr {
	return uintptr(array + uintptr(SIZEOF_CFLOAT*index))
}
