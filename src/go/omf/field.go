//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.


package omf

import (
	"tensor"
)

type Interface interface {
	GetData() (data tensor.Interface, multiplier float32, unit string)
	GetMesh() (cellsize []float32, unit string)
	GetMetadata() map[string]string
}


const (
	X = 0
	Y = 1
	Z = 2
)
