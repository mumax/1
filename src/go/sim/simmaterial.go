//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

// This file implements the methods for setting
// material parameters.

// Sets the exchange constant, defined in J/m
func (s *Sim) AExch(a float) {
	s.input.aexch = a
	s.invalidate()
}

// Sets the saturation magnetization, defined in A/m
func (s *Sim) MSat(ms float) {
	s.input.msat = ms
	s.invalidate()
}

// Sets the damping coefficient
func (s *Sim) Alpha(a float) {
	s.input.alpha = a
	s.invalidate()
}
