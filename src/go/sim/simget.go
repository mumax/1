//  This file is part of MuMax, a high-performance micromagnetic simulator
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import (
	"fmt"
)

func Send(v ...interface{}) {
	fmt.Print("%")
	fmt.Println(v...)
}

func (s *Sim) GetM(component int) {
	Send(s.getM(2 - component)) // translate to ZYX

}
func (s *Sim) getM(component int) float32 {
	s.init()
	return s.devsum.Reduce(s.mDev.comp[component]) / s.avgNorm
}
