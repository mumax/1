//  This file is part of MuMax, a high-performance micromagnetic simulator
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package main

import (
	. "mumax/common"
	"fmt"
)

// Gets a description tag from the table headers and
// appends the values to a column with the name of that tag.
func GetDesc(key string) {
	desc := table.Desc[key]
	newtable.EnsureColumn(key, "")
	value := Atof32(fmt.Sprint(desc))
	newtable.AppendToColumn(key, value)
}
