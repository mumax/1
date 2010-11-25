//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

// The Geom interface describes arbitrary geometries

import (
	. "math"
)

// Returns true if the point (x,y,z) (in SI units, internal axes)
// lies inside the geometry
type Geom interface {
	Inside(x, y, z float32) bool
}

//////////// csg operations

// Union (boolean OR) of geometries
type Or struct {
	children []Geom
}

func (s *Or) Inside(x, y, z float32) bool {
	for _, child := range s.children {
		if child.Inside(x, y, z) {
			return true
		}
	}
	return false
}


// Intersection (boolean AND) of geometries
type And struct {
	children []Geom
}

func (s *And) Inside(x, y, z float32) bool {
	for _, child := range s.children {
		if !child.Inside(x, y, z) {
			return false
		}
	}
	return true
}


///////////// affine transforms

// Translates the wrapped geometry
type Translated struct {
	original               Geom
	deltaX, deltaY, deltaZ float32
}

func (s *Translated) Inside(x, y, z float32) bool {
	return s.original.Inside(x-s.deltaX, y-s.deltaY, z-s.deltaZ)
}


// Ellipsoid with semi-axes rx, ry, rz.
// Becomes a cylinder when an axis is infinte.
type Ellipsoid struct {
	rx, ry, rz float32
}

func (s *Ellipsoid) Inside(x, y, z float32) bool {
	x /= s.rx
	y /= s.ry
	z /= s.rz
	return x*x+y*y+z*z <= 1.
}


// DEBUG
type Wave struct {
	w, h float32
}

func (w *Wave) Inside(x, y, z float32) bool {
	h := w.h / 8
	sin := float32(Sin(float64(z / w.w * Pi)))
	return x < h*(sin+3) && x > h*(sin-3)
}
