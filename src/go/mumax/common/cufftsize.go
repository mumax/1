//  This file is part of MuMax, a high-perfomrance micromagnetic simulator.
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

//  Generate Finite Diference grid sizes that are well suited for CUFFT

package common

// Checks if size = 2^n * {1,3,5,7},
// which is very suited as CUFFT transform size.
func IsGoodCUFFTSize(n int) bool {
	if n < 1 {
		return false
	}
	for n%2 == 0 {
		n /= 2
	}
	if n%3 == 0 {
		n /= 3
	}
	if n%5 == 0 {
		n /= 5
	}
	if n%7 == 0 {
		n /= 7
	}
	return n == 1
}
