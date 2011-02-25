//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

// Some public functions for debugging purposes


// Invalidates the simuation state
func (s *Sim) Invalidate() {
	s.invalidate()
}

// Saves H to file
func (s *Sim) SaveH(fname, format string) {
	s.init()
	s.calcHeff(s.mDev, s.hDev)
	TensorCopyFrom(s.hDev, s.hLocal)
	s.saveOmf(s.hLocal, fname, "Msat", format)
}

// Saves m to file
func (s *Sim) SaveM(fname, format string) {
	s.init()
	TensorCopyFrom(s.mDev, s.mLocal)
	s.saveOmf(s.mLocal, fname, "Msat", format)
}


// Overrides whether the magnetostatic field should be calculated.
// false disables the entire convolution.
// If not, the exchange should be calculated separately.
func (s *Sim) Demag(calc_demag bool) {
	s.input.wantDemag = calc_demag
}
