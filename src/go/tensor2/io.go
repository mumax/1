//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package tensor2

import (
	"io"
	"os"
	"fmt"
	"bufio"
	"encoding/binary"
	"iotool"
	"strings"
)

// Header tokens
const (
	H_COMMENT   = "#"
	H_SEPARATOR = ":"
	H_FORMAT    = "tensorformat"
	H_RANK      = "rank"
	H_SIZE      = "size"
	H_END       = "data"
)

// Central definition of our machine's endianess
var ENDIANESS = binary.LittleEndian

// Writes tensor header (format version number, rank, size) 
func WriteHeader(out_ io.Writer, t Interface) {
	out := bufio.NewWriter(out_)
	defer out.Flush()

	fmt.Fprintln(out, H_COMMENT, H_FORMAT, H_SEPARATOR, 1)
	fmt.Fprintln(out, H_COMMENT, H_RANK, H_SEPARATOR, Rank(t))
	for i, s := range t.Size() {
		fmt.Fprintln(out, H_COMMENT, H_SIZE+fmt.Sprint(i), H_SEPARATOR, s)
	}
}

func ReadHeader(in_ io.Reader) map[string]string {
  header := make(map[string]string)
	in := bufio.NewReader(in_)
	line, eof := iotool.ReadLine(in)
	for !eof {
		key, value := parseHeaderLine(line)
 		header[key] = value
		line, eof = iotool.ReadLine(in)
	}
	return header
}

func parseHeaderLine(str string) (key, value string) {
	strs := strings.Split(str, H_SEPARATOR, 2)
	key = strings.Trim(strs[0], "# ")
	value = strings.Trim(strs[1], "# ")
	return
}

// Writes the tensor data as ascii.
// Includes some newlines to make it human-readable
func WriteDataAscii(out_ io.Writer, t Interface) os.Error {
	out := bufio.NewWriter(out_)
	defer out.Flush()

	for i := NewIterator(t); i.HasNext(); i.Next() {
		if _, err := fmt.Fprint(out, i.Get(), "\t"); err != nil {
			return err
		}

		for j := 0; j < Rank(t); j++ {
			newline := true
			for k := j; k < Rank(t); k++ {
				if i.Index()[k] != t.Size()[k]-1 {
					newline = false
				}
			}
			if newline {
				if _, err := fmt.Fprint(out, "\n"); err != nil {
					return err
				}
			}
		}
	}
	return nil //success
}


// Writes the tensor data in binary (32-bit floats)
func WriteDataBinary(out_ io.Writer, t Interface) (err os.Error) {
	list := t.List()
	out := bufio.NewWriter(out_)
	defer out.Flush()
	err = binary.Write(out, ENDIANESS, list)
	return
}


func ReadDataBinary(in_ io.Reader, t Interface) (err os.Error) {
	list := t.List()
	in := bufio.NewReader(in_)
	err = binary.Read(in, ENDIANESS, list)
	return
}


// TEMP HACK
func FReadAscii4(fname string) *T {
	in, err := os.Open(fname, os.O_RDONLY, 0666)
	defer in.Close()
	if err != nil {
		panic(err)
	}
	return ReadAscii4(in)
}


// TEMP HACK: RANK IS NOT STORED IN ASCII FORMAT
// ASSUME 4
func ReadAscii4(in_ io.Reader) *T {
	rank := 4
	//  _, err := fmt.Fscan(in, &rank)
	//  if err != nil {
	//    panic(err)
	//  }

	in := bufio.NewReader(in_)

	size := make([]int, rank)
	for i := range size {
		_, err := fmt.Fscan(in, &size[i])
		if err != nil {
			panic(err)
		}
	}

	t := NewT(size)
	list := t.List()

	for i := range list {
		_, err := fmt.Fscan(in, &list[i])
		if err != nil {
			panic(err)
		}
	}

	return t
}
