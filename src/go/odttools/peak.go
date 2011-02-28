//  This file is part of MuMax, a high-performance micromagnetic simulator
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package main

import ()

// Finds the time when the value in the specified column first crosses
// the threshold value.
// Typical usage:
// odttool --peak=torque,100,time *.omf
func Peak(column string, threshold float32, output string) {
	col := table.GetColumn(column)
	time := table.GetColumn(output)
	newtable.EnsureColumn(output+"_peak", table.GetUnit(column))

	peakpos := 0
	var value float32 = 0.
	for peakpos = range col {
		if col[peakpos] > threshold {
			value = time[peakpos]
			break
		}
	}
	newtable.AppendToColumn(output+"_peak", value)
}
