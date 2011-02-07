//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

// This file implements the methods for setting
// spintorque parameters.

import (
	. "mumax/common"
)

// Sets the degree of non-adiabaticity (dimensionless)
func (s *Sim) Xi(xi float32) {
	s.xi = xi
}

// Sets the current spin polarization (0-100%)
func (s *Sim) SpinPolarization(p float32) {
	s.spinPol = p
}

// Sets the current density (A/m^2)
func (s *Sim) CurrentDensity(jz, jy, jx float32) {
	s.input.j[X] = jx
	s.input.j[Y] = jy
	s.input.j[Z] = jz
	s.Println("current density: ", s.input.j, "A/m^2")
}
