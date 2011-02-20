//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package tensor

// Functions to read tensors as binary data. 
// Intended for fast inter-process communication or data caching,
// not as a user-friendly format to store simulation output (use mumax/omf for that).
// Uses the machine's endianess.

import (
	. "mumax/common"
	"io"
	"bufio"
	"unsafe"
)

// Utility to read from a file instead of io.Reader
func ReadF(filename string) *T {
	in := MustOpenRDONLY(filename)
	defer in.Close()
	buf := bufio.NewReader(in)
	return Read(buf)
}


func Read(in_ io.Reader) *T {
	in := NewBlockingReader(in_) // Do not read incomplete slices
	size := readHeader(in)
	t := NewT(size)
	readData(in, t.TList)
	return t
}


func readHeader(in *BlockingReader) (size []int){
	var bytes4 [4]byte
	bytes := bytes4[:]
	in.Read(bytes)
	magic := BytesToInt(&bytes4)
	if magic != T_MAGIC {
		panic(IOErr("Bad tensor header: " + string(bytes)))
	}
	in.Read(bytes)
	rank := BytesToInt(&bytes4)
	size = make([]int, rank)
	for i := range size {
		in.Read(bytes)
		size[i] = BytesToInt(&bytes4)
	}
	return
}

func readData(in *BlockingReader, data []float32){
	var bytes4 [4]byte
	bytes := bytes4[:]
	for i := range data {
		in.Read(bytes)
		data[i] = BytesToFloat(&bytes4)
	}
}

// Converts the raw int data to a slice of 4 bytes
func BytesToInt(bytes *[4]byte) int {
	return *((*int)(unsafe.Pointer(bytes)))
}

// Converts the raw float data to a slice of 4 bytes
func BytesToFloat(bytes *[4]byte) float32 {
	return *((*float32)(unsafe.Pointer(bytes)))
}


// Reads data from the reader to the
// (already allocated) tensor.
func (t *T) ReadFrom(in_ io.Reader) {
	in := NewBlockingReader(in_) // Do not read incomplete slices
	size := readHeader(in)
	Assert(EqualSize(size, t.Size()))
	readData(in, t.List())
}


// Reads data from the named file to the
// (already allocated) tensor.
func (t *T) ReadFromF(filename string){
	in := MustOpenRDONLY(filename)
	defer in.Close()
	buf := bufio.NewReader(in)
	t.ReadFrom(buf)
}

