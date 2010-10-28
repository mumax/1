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
	aExch  float32 // Exchange constant in J/m
	mSat   float32 // Saturation magnetization in A/m
	mu0    float32 // mu0 in N/A^2
	gamma0 float32 // Gyromagnetic ratio in m/As
	alpha  float32 // Damping parameter
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


//  Units:
//  FIELD = Ms
//  LENGTH = sqrt(2.0*A/(mu0*Ms*Ms))
//  TIME = 1.0 / (gamma * Ms)
//  ENERGY = A * LENGTH

// The internal unit of length, expressed in meters.
func (mat *Material) UnitLength() float32 {
	assert(mat.Valid())
	return float32(Sqrt(2. * float64(mat.aExch/(mat.mu0*mat.mSat*mat.mSat))))
}


// The internal unit of time, expressed in seconds.
func (mat *Material) UnitTime() float32 {
	assert(mat.Valid())
	return 1.0 / (mat.gamma0 * mat.mSat)
}


// The internal unit of field, expressed in tesla.
func (mat *Material) UnitField() float32 {
	assert(mat.Valid())
	return mat.mu0 * mat.mSat
}


// The internal unit of energy, expressed in J.
func (mat *Material) UnitEnergy() float32 {
	assert(mat.Valid())
	return mat.aExch * mat.UnitLength()
}

// The internal unit of electrical current, expressed in A
func (mat *Material) UnitCurrent() float32 {
  assert(mat.Valid())
  return mat.mSat * mat.UnitLength()
}

// The internal unit of electrical charge, expressed in Q
func (mat *Material) UnitCharge() float32 {
  assert(mat.Valid())
  return mat.UnitCurrent() * mat.UnitTime()
}


// Returns true if the material parameters are valid
func (mat *Material) Valid() bool {
	return mat.aExch > 0. && mat.mSat > 0. && mat.gamma0 > 0 && mat.mu0 > 0
}


func (unit *Material) AssertValid() {
	assert(unit.Valid())
}
