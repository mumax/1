//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package tensor

// This file implements a 3-component vector that implements tensor.Interface.

import (
	. "math"
)

// A 3-component vector that implements tensor.Interface
type Vector [3]float32

func (v *Vector) Size() []int {
	return []int{3}
}

func (v *Vector) List() []float32 {
	return v[0:]
}


func NewVector() *Vector {
	return new(Vector)
}

func UnitVector(direction int) *Vector {
	v := NewVector()
	v[direction] = 1.
	return v
}

func (v *Vector) Set(x, y, z float32) {
	v[0] = x
	v[1] = y
	v[2] = z
}

func (v *Vector) SetTo(other *Vector) {
	v[0] = other[0]
	v[1] = other[1]
	v[2] = other[2]
}

func (a *Vector) Cross(b *Vector) Vector {
	var cross Vector
	cross[0] = a[1]*b[2] - a[2]*b[1]
	cross[1] = a[0]*b[2] - a[2]*b[0]
	cross[2] = a[0]*b[1] - a[1]*b[0]
	return cross
}

func (a *Vector) Dot(b *Vector) float32 {
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
}

func (v *Vector) Norm() float32 {
	return float32(Sqrt(float64(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])))
}

func (v *Vector) Normalize() {
	invnorm := 1. / v.Norm()
	v[0] *= invnorm
	v[1] *= invnorm
	v[2] *= invnorm
}

func (v *Vector) Scale(r float32) {
	v[0] *= r
	v[1] *= r
	v[2] *= r
}

func (v *Vector) Divide(r float32) {
	v[0] /= r
	v[1] /= r
	v[2] /= r
}

func (v *Vector) Sub(other *Vector) {
	v[0] -= other[0]
	v[1] -= other[1]
	v[2] -= other[2]
}

func (v *Vector) Add(other *Vector) {
	v[0] += other[0]
	v[1] += other[1]
	v[2] += other[2]
}
