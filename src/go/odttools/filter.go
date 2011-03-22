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
	"os"
	"math"
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


// Like Matrix() but outputs octave/matlab commands that set
// x, y and data for use in surf(x,y,data)
func Meshdom(i_colname, j_colname, data_colname string) {
	matrix(i_colname, j_colname, data_colname, true)
}


// Assuming columns i,j contain matrix indices,
// coutput column data in a correspondig 2D grid.
// Missing values become 0.
func Matrix(i_colname, j_colname, data_colname string, octave_format bool) {
	matrix(i_colname, j_colname, data_colname, false)
}

func matrix(i_colname, j_colname, data_colname string, octave_format bool) {
	i_col := table.GetColumnIndex(i_colname)
	j_col := table.GetColumnIndex(j_colname)
	data_col := table.GetColumnIndex(data_colname)

	// (1) Construct a sorted set of unique i,j indices (floats).
	// This is the "meshdom", in matlab terms.
	I := table.Column[i_col]
	setI := MakeSet()
	for i := range I {
		setI.Add(I[i])
	}
	I = setI.ToArray()

	setJ := MakeSet()
	J := table.Column[j_col]
	for i := range J {
		setJ.Add(J[i])
	}
	J = setJ.ToArray()

	if octave_format {
		fmt.Print("x=[")
		for i := range I {
			if i != 0 {
				fmt.Print(", ")
			}
			fmt.Print(I[i])
		}
		fmt.Println("];")

		fmt.Print("y=[")
		for i := range J {
			if i != 0 {
				fmt.Print(", ")
			}
			fmt.Print(J[i])
		}
		fmt.Println("];")
	}

	var SENTINEL float32 = -123.456789 // quick and dirty hack

	// (2) Make the "outer product" of the two index sets,
	// spanning a matrix that can be index with each possible i,j pair
	// (even those not present in the input, their data will be 0.)
	matrix := make(map[float32]map[float32]float32)
	for i := range I {
		for j := range J {
			if matrix[I[i]] == nil {
				matrix[I[i]] = make(map[float32]float32)
			}
			matrix[I[i]][J[j]] = SENTINEL
		}
	}

	// (3) Loop over the i indices in the output and add the corrsponing data
	// to the corresponding i,j position of the matrix. (j, data on the same line as i)
	// Missing pairs keep 0. as data.
	D := table.Column[data_col]
	for i := range table.Column[i_col] {
		matrix[table.Column[i_col][i]][table.Column[j_col][i]] = D[i]
	}

	// (3.5)
	// Missing data gets replaced by nearest value
	DELTA := 2 // do not look further than DELTA neighbors 
	for i := range I {
		for j := range J {
			if matrix[I[i]][J[j]] == SENTINEL {

				fmt.Fprintln(os.Stderr, "missing: ", i, j)
				minDst := float32(math.Inf(1))
				nearest := float32(0)
				for i_ := imax(0, i-DELTA); i_ < imin(len(I), i+DELTA); i_++ {
					for j_ := imax(0, j-DELTA); j_ < imin(len(J), j+DELTA); j_++ {
						if matrix[I[i_]][J[j_]] != SENTINEL {

							dst := sqr(I[i]-I[i_]) + sqr(J[j]-J[j_])
							if dst < minDst {
								minDst = dst
								nearest = matrix[I[i_]][J[j_]]
							}

						}
					}
				}
				matrix[I[i]][J[j]] = nearest

			}
		}
	}

	//(4) Print the matrix
	if octave_format {
		fmt.Print("data=reshape([")
	}
	for ind_i, i := range I {
		for ind_j, j := range J {
			fmt.Print(matrix[i][j], "\t")
			if octave_format && !(ind_i == len(I)-1 && ind_j == len(J)-1) {
				fmt.Print(",")
			}
		}
		if !octave_format {
			fmt.Println()
		}
	}
	if octave_format {
		fmt.Println("], ", len(J), ", ", len(I), ");")
	}
	haveOutput = true
}


func imin(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func imax(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func sqr(x float32) float32 {
	return x * x
}
