//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package tensor

import (
	"io"
	"os"
	"bufio"
	"fmt"
	"encoding/binary"
	"iotool"
	"strings"
	"strconv"
)

// Utility to read from a file instead of io.Reader
func ReadF(filename string) *T {
	in, err := os.Open(filename, os.O_RDONLY, 0666)
	defer in.Close()
	if err != nil {
		panic(err)
	}
	return Read(in)
}

func Read(in_ io.Reader) *T {
	in := bufio.NewReader(in_)
	metadata := ReadHeader(in)
	size := metaGetSize(metadata)
	t := NewT(size)
	binary := metaGetBool(metadata, H_BINARY)
	if binary {
		ReadDataBinary(in, t)
	} else {
		ReadDataAscii(in, t)
	}
	return t
}

// Reads data from the reader to the
// (already allocated) tensor.
// NOTE: implements io.ReaderFrom
func (t *T) ReadFrom(in_ io.Reader) {
	in := bufio.NewReader(in_)
	metadata := ReadHeader(in)
	size := metaGetSize(metadata)
	for i, s := range size {
		if s != t.TSize[i] {
			panic("illegal argument: size mismatch")
		}
	}
	binary := metaGetBool(metadata, H_BINARY)
	if binary {
		ReadDataBinary(in, t)
	} else {
		ReadDataAscii(in, t)
	}
}

// INTERNAL
// gets the rank/size from tensor metadata
func metaGetSize(metadata map[string]string) []int {
	rank := metaGetInt(metadata, H_RANK)
	size := make([]int, rank)
	for i := range size {
		size[i] = metaGetInt(metadata, H_SIZE+fmt.Sprint(i))
	}
	return size
}

func metaGetInt(metadata map[string]string, key string) int {
	i, err := strconv.Atoi(metadata[key])
	if err != nil {
		panic(err)
	}
	return i
}

func metaGetBool(metadata map[string]string, key string) bool {
	b, err := strconv.Atob(metadata[key])
	if err != nil {
		panic(err)
	}
	return b
}

// INTERNAL
// Reads a tensor header.
// Returns a map with the key/value pairs in the header
func ReadHeader(in_ io.Reader) map[string]string {
	header := make(map[string]string)
	in := bufio.NewReader(in_)
	line, eof := iotool.ReadLine(in)
	for !eof && !isHeaderEnd(line) {
		key, value := parseHeaderLine(line)
		header[key] = value
		line, eof = iotool.ReadLine(in)
	}
	return header
}

// INTERNAL: Splits "# key: value" into "key", "value"
func parseHeaderLine(str string) (key, value string) {
	strs := strings.Split(str, H_SEPARATOR, 2)
	key = strings.Trim(strs[0], "# ")
	value = strings.Trim(strs[1], "# ")
	return
}

// INTERNAL: true if line == "# begin_data"
func isHeaderEnd(str string) bool {
	return strings.Trim(str, "# ") == H_END
}


// INTERNAL
func ReadDataBinary(in_ io.Reader, t Interface) {
	in := bufio.NewReader(in_)
	list := t.List()
	err := binary.Read(in, ENDIANESS, list)
	if err != nil {
		panic(err)
	}
}

// INTERNAL
func ReadDataAscii(in_ io.Reader, t Interface) {
	in := bufio.NewReader(in_)
	list := t.List()
	for i := range list {
		_, err := fmt.Fscan(in, &list[i])
		if err != nil {
			panic(err)
		}
	}
}
