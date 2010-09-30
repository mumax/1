//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package tensor

import (
	. "math"
)

type Vector struct {
	Component [3]float
}

/** Implements Tensor interface. */

func (v *Vector) Get(index []int) float {
	return v.Component[index[0]]
}

func (v *Vector) Size() []int {
	return []int{3}
}

/** Vector-specific */

func NewVector() *Vector {
	return &Vector{[3]float{0., 0., 0.}}
}

func UnitVector(direction int) *Vector {
	v := NewVector()
	v.Component[direction] = 1.
	return v
}

func (v *Vector) Set(x, y, z float) {
	v.Component[0] = x
	v.Component[1] = y
	v.Component[2] = z
}

func (v *Vector) SetTo(other *Vector) {
	v.Component[0] = other.Component[0]
	v.Component[1] = other.Component[1]
	v.Component[2] = other.Component[2]
}

func (a *Vector) Dot(b *Vector) float {
	return a.Component[0]*b.Component[0] + a.Component[1]*b.Component[1] + a.Component[2]*b.Component[2]
}

func (v *Vector) Norm() float {
	return float(Sqrt(float64(v.Component[0]*v.Component[0] + v.Component[1]*v.Component[1] + v.Component[2]*v.Component[2])))
}

func (v *Vector) Normalize() {
	invnorm := 1. / v.Norm()
	v.Component[0] *= invnorm
	v.Component[1] *= invnorm
	v.Component[2] *= invnorm
}

func (v *Vector) Scale(r float) {
	v.Component[0] *= r
	v.Component[1] *= r
	v.Component[2] *= r
}

func (v *Vector) Divide(r float) {
	v.Component[0] /= r
	v.Component[1] /= r
	v.Component[2] /= r
}

func (v *Vector) Sub(other *Vector) {
	v.Component[0] -= other.Component[0]
	v.Component[1] -= other.Component[1]
	v.Component[2] -= other.Component[2]
}

func (v *Vector) Add(other *Vector) {
	v.Component[0] += other.Component[0]
	v.Component[1] += other.Component[1]
	v.Component[2] += other.Component[2]
}
