//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package main

import (
	"sim"
	"fmt"
)

// Wrapper for sim.Main.
// This is a bit silly but sim.Main() is in package sim (not package main)
// so it cannot be compiled to an executable directly.
func main() {
  fmt.Println(WELCOME)
	sim.Main()
}


const WELCOME = `
  MuMax 0.4.1514
  (c) Arne Vansteenkiste & Ben Van de Wiele,
      DyNaMat/EELAB UGent
  This version is meant for internal testing purposes only,
  please contact the authors if you like to distribute this program.
  
`