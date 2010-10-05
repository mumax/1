//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import (
	. "math"
	// 	"fmt"
)

type Material struct {
	aExch  float // Exchange constant in J/m
	mSat   float // Saturation magnetization in A/m
	mu0    float // mu0 in N/A^2
	gamma0 float // Gyromagnetic ratio in m/As
	alpha  float // Damping parameter
}


func NewMaterial() *Material {
	mat := new(Material)
	mat.InitMaterial()
	return mat
}

func (mat *Material) InitMaterial() {
	mat.mu0 = 4.0E-7 * Pi
	mat.gamma0 = 2.211E5
}


// func (mat *Material) String() string {
// 	s := "Material:\n"
// 	s += fmt.Sprintln("aExch      : \t", mat.aExch, " J/m")
// 	s += fmt.Sprintln("mSat       : \t", mat.mSat, " A/m")
// 	s += fmt.Sprintln("gamma0     : \t", mat.gamma0, " m/As")
// 	s += fmt.Sprintln("mu0        : \t", mat.mu0, " N/A^2")
// 	s += fmt.Sprintln("exch length: \t", mat.UnitLength(), " m")
// 	s += fmt.Sprintln("unit time  : \t", mat.UnitTime(), " s")
// 	s += fmt.Sprintln("unit energy: \t", mat.UnitEnergy(), " J")
// 	s += fmt.Sprintln("unit field : \t", mat.UnitField(), " T")
// 	return s
// }


//  FIELD = Ms
//  LENGTH = sqrt(2.0*A/(mu0*Ms*Ms))
//  TIME = 1.0 / (gamma * Ms)
//  ENERGY = A * LENGTH

// The internal unit of length, expressed in meters.
func (mat *Material) UnitLength() float {
	assert(mat.Valid())
	return float(Sqrt(2. * float64(mat.aExch/(mat.mu0*mat.mSat*mat.mSat))))
}


// The internal unit of time, expressed in seconds.
func (mat *Material) UnitTime() float {
	assert(mat.Valid())
	return 1.0 / (mat.gamma0 * mat.mSat)
}


// The internal unit of field, expressed in tesla.
func (mat *Material) UnitField() float {
	assert(mat.Valid())
	return mat.mu0 * mat.mSat
}


// The internal unit of energy, expressed in J.
func (mat *Material) UnitEnergy() float {
	assert(mat.Valid())
	return mat.aExch * mat.UnitLength()
}


// Returns true if the material parameters are valid
func (mat *Material) Valid() bool {
	return mat.aExch > 0. && mat.mSat > 0. && mat.gamma0 > 0 && mat.mu0 > 0
}


func (unit *Material) AssertValid() {
	assert(unit.Valid())
}
