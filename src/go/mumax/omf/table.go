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
	PrintHeader      bool
}


func (t *Table) Init() {
	t.PrintHeader = true
	t.Desc = make(map[string]interface{})
}


// Retrieve a column index identified by name, e.g., "Mx".
// Used to index Table.Column
// Returns -1 if the column does not exist.
func (t *Table) GetColumnIndex(name string) int {
	name = ToLower(name)
	for i, n := range t.ColName {
		if ToLower(n) == name {
			return i
		}
	}
	return -1
}

// Retrieve a column identified by name, e.g., "Mx".
// Returns nil if the column does not exist.
func (t *Table) GetColumn(name string) []float32 {
	index := t.GetColumnIndex(name)
	if index < 0 {
		return nil
	}
	return t.Column[index]
}


// Retrieve the unit of a column identified by name, e.g., "Mx".
// Returns "" if the column does not exist.
func (t *Table) GetUnit(name string) string {
	index := t.GetColumnIndex(name)
	if index < 0 {
		return ""
	}
	return t.ColUnit[index]
}


// Add a column with specified name and unit.
func (t *Table) AddColumn(name, unit string) {
	if t.GetColumn(name) != nil {
		panic(Bug("omf.Table: column already exists: " + name))
	}
	t.ColName = append(t.ColName, name)
	t.ColUnit = append(t.ColUnit, unit)
	t.Column = append(t.Column, make([]float32, 0, 100))
}


// Ensure a column with specified name is present and retrieve it.
// Adds the column only when it did not yet exist.
func (t *Table) EnsureColumn(name, unit string) []float32 {
	col := t.GetColumn(name)
	if col == nil {
		t.AddColumn(name, unit)
		col = t.GetColumn(name)
	}
	return col
}


// Append data to the column specified by name.
func (t *Table) AppendToColumn(name string, value float32) {
	index := t.GetColumnIndex(name)
	t.Column[index] = append(t.Column[index], value)
}


// Write table to output stream.
func (t *Table) WriteTo(out io.Writer) {
	writer := NewTabWriter(out)
	writer.PrintHeader = t.PrintHeader
	for i := range t.ColName {
		writer.AddColumn(t.ColName[i], t.ColUnit[i])
	}
	for row := range t.Column[0] {
		for i := range t.Column {
			writer.Print(t.Column[i][row])
		}
	}
	writer.Close()
}


// Reads a table
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
			var value float32	
			n, _ = fmt.Fscan(in, &value)
			if n != 0{
				t.Column[i] = append(t.Column[i], value) //TODO: not very efficient but there is no containter/Float32Vector...
			}
		}
		row++
	}
	return t
}

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
