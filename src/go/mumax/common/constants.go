//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package common

// Central definition of common constants.


// direction flag for memcpy()
const (
	CPY_TO   = 1
	CPY_ON   = 2
	CPY_FROM = 3
)

// direction flag for copyPadded()
const (
	CPY_PAD   = 1
	CPY_UNPAD = 2
)

// direction flag for FFT
const (
	FFT_FORWARD = 1
	FFT_INVERSE = -1
)

// Reduction operation flags for reduce()
const (
	ADD    = 1
	MAX    = 2
	MAXABS = 3
	MIN    = 4
)
