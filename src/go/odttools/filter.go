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

type Empty struct{}

type Set map[float32]Empty

func MakeSet() Set {
	return Set(make(map[float32]Empty))
}

func (s Set) Add(x float32) {
	if _, ok := s[x]; !ok {
		s[x] = Empty{}
	}
}

func (s Set) ToArray() []float32 {
	array := make([]float32, len(s))
	i := 0
	for val, _ := range s {
		array[i] = val
		i++
	}
	Float32Array(array).Sort()
	return array
}

// Assuming columns i,j contain matrix indices,
// coutput column data in a correspondig 2D grid.
// Missing values become 0.
func Matrix(i_col, j_col, data_col int) {

	// (1) Construct a sorted set of unique i,j indices (floats)
	I := table.Column[i_col]
	fmt.Println(I)
	setI := MakeSet()
	for i := range I {
		setI.Add(I[i])
	}
	I = setI.ToArray()
	fmt.Println(setI)

	setJ := MakeSet()
	J := table.Column[j_col]
	fmt.Println(J)
	for i := range J {
		setJ.Add(J[i])
	}
	J = setJ.ToArray()
	fmt.Println(setJ)

	// (2) Make the "outer product" of the two index sets,
	// spanning a matrix that can be index with each possible i,j pair
	// (even those not present in the input, their data will be 0.)
	matrix := make(map[float32]map[float32]float32)
	for i := range I {
		for j := range J {
			if matrix[I[i]] == nil {
				matrix[I[i]] = make(map[float32]float32)
			}
			matrix[I[i]][J[j]] = 0.
		}
	}

	// (3) Loop over the i indices in the output and add the corrsponing data
	// to the corresponding i,j position of the matrix. (j, data on the same line as i)
	// Missing pairs keep 0. as data.
	D := table.Column[data_col]
	for i := range I {
		matrix[I[i]][J[i]] = D[i]
	}

	// (4) Print the matrix
	for _, col := range matrix {
		for _, data := range col {
			fmt.Print(data, "\t")
		}
		fmt.Println()
	}
	haveOutput=true
}
