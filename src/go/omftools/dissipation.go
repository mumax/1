//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package main


import (
	"mumax/tensor"
	"fmt"
	"mumax/omf"
	. "strings"
)


func Dissipation(ordinate string) {
	hfile := Replace(filename, "m", "h", 1)
	hinfo, hdata := omf.FRead(hfile)
	alpha := hinfo.DescGetFloat32("alpha")
	alpha = 1
	x := info.DescGet(ordinate)

	mx, my, mz := tensor.Component(data, 0).List(), tensor.Component(data, 2).List(), tensor.Component(data, 2).List()
	hx, hy, hz := tensor.Component(hdata, 0).List(), tensor.Component(hdata, 2).List(), tensor.Component(hdata, 2).List()

	sum := 0.	
	for i := range mx{
		var m *tensor.Vector = new(tensor.Vector)
		var h *tensor.Vector = new(tensor.Vector)
		var mxh *tensor.Vector = new(tensor.Vector)
		var mxmxh *tensor.Vector = new(tensor.Vector)

		*m = tensor.Vector([3]float32{mx[i], my[i], mz[i]})
		*h = tensor.Vector([3]float32{hx[i], hy[i], hz[i]})
		*mxh = m.Cross(h)
		*mxmxh = m.Cross(mxh)
		sum -= float64(alpha * mxmxh.Norm())
	} 


	fmt.Println(x, "\t", sum)
}

