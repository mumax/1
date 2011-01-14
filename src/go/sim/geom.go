//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

// The Geom interface describes arbitrary geometries

import (
	. "math"
	"image/png"
	"os"
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

type Inverse struct {
	original Geom
}

func (s *Inverse) Inside(x, y, z float32) bool {
	return !s.original.Inside(x, y, z)
}

type Array struct {
	original Geom
	nz, ny   int
	dz, dy   float32
}

func (a *Array) Inside(x, y, z float32) bool {
	for i := 0; i < a.ny; i++ {
		for j := 0; j < a.nz; j++ {
			ty := y - (a.dy * (float32(i) - float32(a.ny)/2. + .5))
			tz := z - (a.dz * (float32(j) - float32(a.nz)/2. + .5))
			if a.original.Inside(x, ty, tz) {
				return true
			}
		}
	}
	return false
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

type Cuboid Ellipsoid

func (s *Cuboid) Inside(x, y, z float32) bool {
	x /= s.rx
	y /= s.ry
	z /= s.rz
	return abs32(x) <= 1. && abs32(y) <= 1. && abs32(z) <= 1.
}


type Mask struct {
	inside       [][]bool
	sizey, sizez float32
}


func NewMask(fname string, sizey, sizez float32) *Mask {
	in, err := os.Open(fname, os.O_RDONLY, 0666)
	if err != nil {
		panic(err)
	}

	img, err2 := png.Decode(in)
	if err2 != nil {
		panic(err2)
	}

	width := img.Bounds().Max.X
	height := img.Bounds().Max.Y

	inside := make([][]bool, height)
	for i := range inside {
		inside[i] = make([]bool, width)
	}

	for i := range inside {
		for j := range inside[i] {
			r, g, b, _ := img.At(j, height-1-i).RGBA()
			if r+g+b < (0xFFFF*3)/2 {
				inside[i][j] = true
			}
		}
	}
	return &Mask{inside, sizey, sizez}
}

func (im *Mask) Inside(x, y, z float32) bool {
	inside := im.inside
	width, height := len(inside[0]), len(inside)

	i := int((y/im.sizey+.5)*float32(height) + .5)
	j := int((z/im.sizez+.5)*float32(width) + .5)

	if i >= 0 && i < height && j >= 0 && j < width {
		return inside[i][j]
	}
	return false
}


func (sim *Sim) Mask(image string) {
	sim.initSize()
	sim.geom = NewMask(image, sim.input.partSize[Y], sim.input.partSize[Z])
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
