//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import (
	"os"
	"fmt"
)

// crashes if the test is false
func assert(test bool) {
	if !test {
		panic("assertion failed")
	}
}


// puts a 3 in front of the array
func Size4D(size3D []int) []int {
	assert(len(size3D) == 3)
	size4D := make([]int, 4)
	size4D[0] = 3
	for i := range size3D {
		size4D[i+1] = size3D[i]
	}
	return size4D
}


// removes the 3 in front of the array
func Size3D(size4D []int) []int {
	assert(len(size4D) == 4)
	assert(size4D[0] == 3)
	size3D := make([]int, 3)
	for i := range size3D {
		size3D[i] = size4D[i+1]
	}
	return size3D
}

// Product of integers.
// Used to get the total number of elements from a tensor size.
func prod(size []int) int {
	prod := 1
	for _, s := range size {
		prod *= s
	}
	return prod
}

var Verbosity int = 2


// func Debug(msg ...interface{}) {
// 	if Verbosity > 0 {
// 		fmt.Fprint(os.Stderr, msg)
// 		fmt.Fprint(os.Stderr, ERASE) // Erase rest of line
// 		fmt.Fprintln(os.Stderr)
// 	}
// }
// 
// func Debugv(msg ...interface{}) {
// 	if Verbosity > 1 {
// 		Debug(msg)
// 	}
// }


func Debugvv(msg ...interface{}) {
	if Verbosity > 2 {
		fmt.Fprint(os.Stderr, msg)
		fmt.Fprint(os.Stderr, ERASE) // Erase rest of line
		fmt.Fprintln(os.Stderr)
	}
}
