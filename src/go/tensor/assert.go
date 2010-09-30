//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package tensor

import "log"

//TODO: move to util?

/** Crashes the program when the test is false. */
func assert(test bool) {
	if !test {
		log.Crash("Assertion failed")
	}
}

/** Crashes the program with an error message when the test is false. */
func assertMsg(test bool, msg string) {
	if !test {
		log.Crash(msg)
	}
}
