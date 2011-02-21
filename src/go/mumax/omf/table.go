//  This file is part of MuMax, a high-performance micromagnetic simulator
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package omf


// This file implements the omf.Table structure that stores an odt table.

import (
	. "mumax/common"
	. "strings"
	"io"
)



type Table struct {
	Desc             map[string]interface{}
	ColName, ColUnit []string
	Column           []float32
}


func ReadTable(in io.Reader) *Table {
	t := new(Table)

	line, eof := ReadLine(in)
	for !eof && !isTableHeaderEnd(line) {
		entries := parseTableHeaderLine(line)

		switch ToLower(entries[0]) {
		default:
			panic(InputErr("Unknown ODT key: " + entries[0]))

		}

		line, eof = ReadLine(in)
	}
	return t
}


// INTERNAL: Does str mark the end of the table header?
func isTableHeaderEnd(str string) bool {
	str = ToLower(Trim(str, "# "))
	str = Replace(str, " ", "", -1)
	return HasPrefix(str, "TableStart")
}


// INTERNAL: Splits "# key: value1 value2 ..." into "key", "value1", "value2", ...
// Values are optional, many entries only have the key.
func parseTableHeaderLine(str string) (split []string) {
	strs := Split(str, ": \t", -1)
	split = make([]string, len(strs))
	split[0] = Trim(strs[0], "# \t")
	for i := range strs {
		split[i] = Trim(strs[0], " \t")
	}
	return
}
