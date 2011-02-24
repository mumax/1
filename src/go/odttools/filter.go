//  This file is part of MuMax, a high-performance micromagnetic simulator
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.


package main


import (
	//. "mumax/common"
)


// --header=false: do not output odt header and footer
func Header(printHeader bool) {
	newtable.PrintHeader = printHeader
}


// add read table to internal table
func Cat() {
	for i := range table.Column {
		newtable.EnsureColumn(table.ColName[i], table.ColUnit[i])
		in := table.Column[i]
		for j := range in {
			newtable.AppendToColumn(table.ColName[i], in[j])
		}
	}
}

// add one column to the internal table
func GetCol(name string) {
	i := table.GetColumnIndex(name)
	newtable.EnsureColumn(table.ColName[i], table.ColUnit[i])
	in := table.Column[i]
	for j := range in {
		newtable.AppendToColumn(table.ColName[i], in[j])
	}
}

// Assuming columns i,j contain matrix indices,
// coutput column data in a correspondig 2D grid.
// Missing values become 0.
func Matrix(i, j, data int){
//	// Sorted copies of the index columns
//	I := make([]float32, len(table.Column[i]))
//	copy(I, table.Column[i])
//	Float32Array(I).Sort()
//	J := make([]float32, len(table.Column[j]))
//	copy(J, table.Column[j])
//	Float32Array(J).Sort()
//
//	
//	D := table.Column[data]
//
//	// Count indices
}
