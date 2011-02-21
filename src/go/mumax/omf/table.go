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
	"fmt"
)


type Table struct {
	Desc             map[string]interface{}
	ColName, ColUnit []string
	Column           [][]float32
}


func ReadTable(in io.Reader) *Table {
	t := new(Table)

	// Read Header
	line, eof := ReadLine(in)
	done := false
	for !eof && !done {
		entries := parseTableHeaderLine(line)

		switch ToLower(entries[0]) {
		default:
			panic(InputErr("Unknown ODT key: " + entries[0]))
		//ignored:
		case "odt", "begin", "end", "table", "title":
		case "desc":
			if t.Desc == nil {
				t.Desc = make(map[string]interface{})
			}
			t.Desc[entries[1]] = entries[2]
		case "units":
			for _, e := range entries[1:] {
				t.ColUnit = append(t.ColUnit, Trim(e, "{}"))
			}
		case "columns":
			for _, e := range entries[1:] {
				t.ColName = append(t.ColName, e)
				t.Column = append(t.Column, make([]float32, 0, 100))
			}
			done = true // #Columns is last header entry, break here.
		}
		if !done {
			line, eof = ReadLine(in)
		}
	}

	// Read data
	n := 1
	row := 0
	for n > 0 {
		for i := range t.Column {
			t.Column[i] = append(t.Column[i], 0.) //TODO: not very efficient but there is no containter/Float32Vector...
			n, _ = fmt.Fscan(in, &(t.Column[i][row]))
		}
		row++
	}
	return t
}

// Retrieve a column identified by name, e.g., "Mx".
func (t *Table) GetColumn(name string) []float32 {
	name = ToLower(name)
	for i, n := range t.ColName {
		if ToLower(n) == name {
			return t.Column[i]
		}
	}
	return nil
}

//// INTERNAL: Does str mark the end of the table header?
//func isTableHeaderEnd(str string) bool {
//	str = ToLower(Trim(str, "# "))
//	str = Replace(str, " ", "", -1)
//	return HasPrefix(str, "TableStart")
//}


// INTERNAL: Splits "# key: value1 value2 ..." into "key", "value1", "value2", ...
// Values are optional, many entries only have the key.
func parseTableHeaderLine(str string) (split []string) {
	//fmt.Print("Split ", str, ":")
	if !HasPrefix(str, "#") {
		panic(InputErr("Expected \"#\": " + str))
	}
	str = Trim(str, "#") + " "
	start, stop := 0, 0
	for i, c := range str {
		if c == int(':') || c == int(' ') || c == int('\t') {
			stop = i
			substr := str[start:stop]
			if substr != "" {
				split = append(split, substr)
			}
			start = stop + 1
		}
	}
	if stop <= len(str) && start < stop {
		substr := str[start:stop]
		split = append(split, substr)
	}

	//fmt.Println(split)
	return
}
