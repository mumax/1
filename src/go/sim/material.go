//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import (
	. "math"
	"fmt"
	"os"
)

type Material struct {
	aExch   float32 // Exchange constant in J/m
	mSat    float32 // Saturation magnetization in A/m
	mu0     float32 // mu0 in N/A^2
	gamma0  float32 // Gyromagnetic ratio in m/As
	muB     float32 // Bohr magneton in Am^2
	e       float32 // Electron charge in As
	alpha   float32 // Damping parameter
	xi      float32 // Spin-transfer torque: degree of non-adiabaticity
	spinPol float32 // Spin-transfer torque: spin-polarization of the electrical current (0-100%)
}


// func NewMaterial() *Material {
// 	mat := new(Material)
// 	mat.InitMaterial()
// 	return mat
// }

func (s *Sim) InitMaterial() {
	s.Println("Initializing material parameters")
	s.mu0 = 4.0E-7 * Pi
	s.gamma0 = 2.211E5
	s.muB = 9.2740091523E-24
	s.e = 1.60217646E-19
	s.spinPol = DEFAULT_SPIN_POLARIZATION
	// even when not used, these arrays should be allocated:
	// they may be passed to C even when they are ignored there
	//s.anisK = make([]float32, 1)
	//s.anisAxes = make([]float32, 1)

	if s.input.msat == 0. {
		s.Errorln("Saturation magnetization should first be set. E.g. msat 800E3")
		os.Exit(-6)
	}
	s.mSat = s.input.msat

	if s.input.aexch == 0. {
		s.Errorln("Exchange constant should first be set. E.g. aexch 12E-13")
		os.Exit(-6)
	}
	s.aExch = s.input.aexch

	if s.alpha <= 0. {
		s.Warn("Damping parameter alpha =  ", s.alpha)
	}

	s.desc["msat"] = fmt.Sprint(s.mSat)
	s.desc["aexch"] = fmt.Sprint(s.aExch)
	s.desc["alpha"] = fmt.Sprint(s.alpha)
}


//  Units:
//  FIELD = Ms
//  LENGTH = sqrt(2.0*A/(mu0*Ms*Ms))
//  TIME = 1.0 / (gamma * Ms)
//  ENERGY = A * LENGTH

// The internal unit of length, expressed in meters.
// NOTE: We use the exchange length here, but we could
// have omitted the factor 2. The implication of including
// the factor 2 here is, e.g.:
//  * The factor 2 in the exchange field formulation has to be dropped
//  * The factor 2 in the uniaxial anisotropy formulation has to be dropped
//  * ...
// This makes no difference in the end but should be noted.
//  
func (mat *Material) UnitLength() float32 {
	assert(mat.Valid())
	return float32(Sqrt(2. * float64(mat.aExch/(mat.mu0*mat.mSat*mat.mSat))))
}


// The internal unit of time, expressed in seconds.
func (mat *Material) UnitTime() float32 {
	assert(mat.Valid())
	return 1.0 / (mat.gamma0 * mat.mSat)
}


// The internal unit of field (magnetic induction "B", not "H"), expressed in tesla.
func (mat *Material) UnitField() float32 {
	assert(mat.Valid())
	return mat.mu0 * mat.mSat
}


// The internal unit of energy, expressed in J.
func (mat *Material) UnitEnergy() float32 {
	return mat.aExch * mat.UnitLength()
}

// The internal unit of energy density, expressed in J/m^3.
func (mat *Material) UnitEnergyDensity() float32 {
	return mat.UnitEnergy() / mat.UnitVolume()
}


// The internal unit of electrical current, expressed in A
func (mat *Material) UnitCurrent() float32 {
	return mat.mSat * mat.UnitLength()
}


// The internal unit of electrical current density, expressed in A/m^2
func (mat *Material) UnitCurrentDensity() float32 {
	return mat.mSat / mat.UnitLength()
}


// The internal unit of electrical charge, expressed in Q
func (mat *Material) UnitCharge() float32 {
	return mat.UnitCurrent() * mat.UnitTime()
}


// The internal unit of volume, expressed in m^3
func (mat *Material) UnitVolume() float32 {
	lex := mat.UnitLength()
	return lex * lex * lex
}


// The internal unit of magnetic moment, expressed in Am^2
func (mat *Material) UnitMoment() float32 {
	return mat.mSat * mat.UnitVolume()
}


// Returns true if the material parameters are valid
func (mat *Material) Valid() bool {
	return mat.aExch > 0. && mat.mSat > 0. && mat.gamma0 > 0 && mat.mu0 > 0
}


func (unit *Material) AssertValid() {
	assert(unit.Valid())
}
