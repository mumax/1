//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import (
	"tensor"
	. "image"
	"image/png"
	"math"
	"io"
)


// func main() {
// 	t := Read(os.Stdin)
// 	if Rank(t) != 2 {
// 		panic("tensor.Tensor should have rank 2")
// 	}
// 	PNG(os.Stdout, t)
// }

func PNG(out io.Writer, t tensor.Tensor) {
	err := png.Encode(out, DrawTensor(t))
	if err != nil {
		panic(err)
	}
}

func DrawTensor(t tensor.Tensor) *NRGBA {
	// todo: we need to handle any rank
	for tensor.Rank(t) == 4 {
		t = tensor.Average(t, 1) //average over the thickness
	}

	// 	w, h := t.Size()[0], t.Size()[1]
	// 	img := NewNRGBA(w, h)
	// 	pos := []int{w, h}
	// 	for i := 0; i < w; i++ {
	// 		pos[X] = i
	// 		for j := 0; j < h; j++ {
	// 			pos[Y] = j
	// 			img.Set(i, j, GreyMap(-1., 1., t.Get(pos)))
	// 		}
	// 	}
	// 	return img

	w, h := t.Size()[1], t.Size()[2]
	img := NewNRGBA(w, h)
	pos := []int{0, w, h}
	for i := 0; i < w; i++ {
		pos[1] = i
		for j := 0; j < h; j++ {
			pos[2] = j
			pos[0] = X
			x := t.Get(pos)
			pos[0] = Y
			y := t.Get(pos)
			pos[0] = Z
			z := t.Get(pos)
			img.Set(i, j, HSLMap(z, y, x)) // TODO: x is thickness for now...
		}
	}
	return img

}

func GreyMap(min, max, value float) NRGBAColor {
	color := (value - min) / (max - min)
	if color > 1. {
		color = 1.
	}
	if color < 0. {
		color = 0.
	}
	color8 := uint8(255 * color)
	return NRGBAColor{color8, color8, color8, 255}
}

func HSLMap(x, y, z float) NRGBAColor {
	s := fsqrt(x*x + y*y + z*z)
	l := 0.5*z + 0.5
	h := float(math.Atan2(float64(x), float64(y)))
	return HSL(h, s, l)
}

func fsqrt(number float) float {
	return float(math.Sqrt(float64(number)))
}

// h = 0..2pi, s=0..1, l=0..1
func HSL(h, s, l float) NRGBAColor {
	if s > 1 {
		s = 1
	}
	if l > 1 {
		l = 1
	}
	for h < 0 {
		h += 2 * math.Pi
	}
	for h > 2*math.Pi {
		h -= math.Pi
	}
	h = h * (180.0 / math.Pi / 60.0)

	// chroma
	var c float
	if l <= 0.5 {
		c = 2 * l * s
	} else {
		c = (2 - 2*l) * s
	}

	x := c * (1 - abs(fmod(h, 2)-1))

	var (
		r, g, b float
	)

	switch {
	case 0 <= h && h < 1:
		r, g, b = c, x, 0.
	case 1 <= h && h < 2:
		r, g, b = x, c, 0.
	case 2 <= h && h < 3:
		r, g, b = 0., c, x
	case 3 <= h && h < 4:
		r, g, b = 0, x, c
	case 4 <= h && h < 5:
		r, g, b = x, 0., c
	case 5 <= h && h < 6:
		r, g, b = c, 0., x
	default:
		r, g, b = 0., 0., 0.
	}

	m := l - 0.5*c
	r, g, b = r+m, g+m, b+m
	R, G, B := uint8(255*r), uint8(255*g), uint8(255*b)
	return NRGBAColor{R, G, B, 255}
}

// modulo
func fmod(number, mod float) float {
	for number < mod {
		number += mod
	}
	for number > mod {
		number -= mod
	}
	return number
}

func abs(number float) float {
	if number < 0 {
		return -number
	} // else
	return number
}
