//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package main

import (
	"sim"
)

// Wrapper for sim.Main.
// This is a bit silly but sim.Main() is in package sim (not package main)
// so it cannot be compiled to an executable directly.
func main() {
	sim.Main()
}
