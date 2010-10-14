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
)


// Header tokens
const (
	H_COMMENT   = "#"
	H_SEPARATOR = ":"
	H_FORMAT    = "tensor_version"
	H_RANK      = "rank"
	H_SIZE      = "size"
	H_BINARY    = "binary"
	H_END       = "begin_data"
)

// Central definition of our machine's endianess
var ENDIANESS = binary.LittleEndian

// TODO: need better error returning,
// also necessary to implement io.WriterTo, ReaderFrom
func (t *T) WriteTo(out io.Writer){
  Write(out, t)
}

// Writes in the default format (binary)
func Write(out_ io.Writer, t Interface){
  WriteBinary(out_, t)
}

// Writes the tensor in binary format.
func WriteBinary(out_ io.Writer, t Interface) {
	WriteMetaTensorBinary(out_, t, nil)
}

// Writes the tensor in ascii format
func WriteAscii(out_ io.Writer, t Interface) {
	WriteMetaTensorAscii(out_, t, nil)
}


// Writes the tensor in binary format,
// plus addional metadata in the form of "key:value" pairs.
// The metadata map may safely be nil.
func WriteMetaTensorBinary(out_ io.Writer, t Interface, metadata map[string]string) {
	out := bufio.NewWriter(out_)
	defer out.Flush()

	WriteTensorHeader(out, t)
	WriteHeaderLine(out, H_BINARY, "true")
	WriteMetaHeader(out, metadata)
	CloseHeader(out)
	WriteDataBinary(out, t)
}

// Writes the tensor in ascii format,
// plus addional metadata in the form of "key:value" pairs.
// The metadata map may safely be nil.
func WriteMetaTensorAscii(out_ io.Writer, t Interface, metadata map[string]string) {
	out := bufio.NewWriter(out_)
	defer out.Flush()

	WriteTensorHeader(out, t)
	WriteHeaderLine(out, "binary", "false")
	WriteMetaHeader(out, metadata)
	CloseHeader(out)
	WriteDataAscii(out, t)
}


// TODO:
// MetaT struct? Tensor + metadata
// Read() (tensor, metadata)
// method T.ReadFrom() implements io.ReaderFrom
// method T.WriteTo()
// 


// INTERNAL
// Writes tensor header (format version number, rank, size),
// except "# begin_data" statement.
// WriteMetaHeader() may be called to add addional header metadata
// before CloseHeader().
// TODO panic on error
func WriteTensorHeader(out_ io.Writer, t Interface) {
	out := bufio.NewWriter(out_)
	defer out.Flush()

	fmt.Fprintln(out, H_COMMENT, H_FORMAT, H_SEPARATOR, 1)
	fmt.Fprintln(out, H_COMMENT, H_RANK, H_SEPARATOR, Rank(t))
	for i, s := range t.Size() {
		fmt.Fprintln(out, H_COMMENT, H_SIZE+fmt.Sprint(i), H_SEPARATOR, s)
	}
}

// INTERNAL
// Writes the content of the map as addional key:value pairs to the tensor header.
// To be called between WriteTensorHeader() and CloseHeader()
func WriteMetaHeader(out_ io.Writer, metadata map[string]string) {
	if metadata == nil {
		return //nothing to do
	}
	for key, val := range metadata {
		WriteHeaderLine(out_, key, val)
	}
}

// INTERNAL
// Closes the header by printing "# begin_data"
func CloseHeader(out_ io.Writer) {
	out := bufio.NewWriter(out_)
	defer out.Flush()
	_, err := fmt.Fprintln(out, H_COMMENT, H_END)
	if err != nil {
		panic(err)
	}
}


// INTERNAL
// Adds a key:value pair to the header.
func WriteHeaderLine(out_ io.Writer, key, value string) {
	out := bufio.NewWriter(out_)
	defer out.Flush()
	fmt.Fprintln(out, H_COMMENT, key, H_SEPARATOR, value)
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
