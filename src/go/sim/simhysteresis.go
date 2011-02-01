//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import (
	"fmt"
)

// Scans the field from (bz0, by0, bx0) to (bz1, by1, bx1) in a number of steps.
// After each field step, the magnetization is saved.
// TODO: output control should be more fine-grained
func (s *Sim) Hysteresis(bz0, by0, bx0, bz1, by1, bx1 float32, steps int) {
	fmt.Fprintf(s.out, "Hysteresis scan from (%f, %f, %f)T to (%f, %f, %f)T in %v steps\n", bx0, by0, bz0, bx1, by1, bz1, steps)
	for i := 0; i <= steps; i++ {
		bx := bx0 + (bx1-bx0)*float32(i)/float32(steps)
		by := by0 + (by1-by0)*float32(i)/float32(steps)
		bz := bz0 + (bz1-bz0)*float32(i)/float32(steps)
		s.StaticField(bz, by, bx)
		s.Relax()
		s.Save("m", "omf")
		s.Save("table", "ascii") 
	}
}

func (s *Sim) RelaxMaxTorque(max float32){
	maxtorque = max
}

