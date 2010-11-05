//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import (
	"fmt"
)

type Geom interface {
	Inside(x, y, z float32) bool
}


type Ellipsoid struct {
	rx, ry, rz float32
}

func (s *Ellipsoid) Inside(x, y, z float32) bool {
	x /= s.rx
	y /= s.ry
	z /= s.rz
	return x*x+y*y+z*z <= 1.
}
