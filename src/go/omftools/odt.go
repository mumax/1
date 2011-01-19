//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package main


import (
	"omf"
	"iotool"
	"tensor"
)

var odt *omf.TabWriter

// Generates a .odt datatable form .omf files.
func ToODT(filename string){
	init_odt(filename)
	odt.Print(info.DescGet("Time"))

	mx := tensor.Average(tensor.Component(data, 0))
	my := tensor.Average(tensor.Component(data, 1))
	mz := tensor.Average(tensor.Component(data, 2))
	odt.Print(mx, my, mz)

	odt.Print(info.DescGet("Bx"))
	odt.Print(info.DescGet("By"))
	odt.Print(info.DescGet("Bz"))
	odt.Print(info.DescGet("id"))
}

func init_odt(filename string){
	if odt == nil{
		out := iotool.MustOpenWRONLY(filename)
		odt = omf.NewTabWriter(out)
		odt.AddColumn("Time", "s")
		odt.AddColumn("Mx/Ms", "")
		odt.AddColumn("My/Ms", "")
		odt.AddColumn("Mz/Ms", "")
		odt.AddColumn("Bx", "T")
		odt.AddColumn("By", "T")
		odt.AddColumn("Bz", "T")
		odt.AddColumn("id", "")
	}
}
