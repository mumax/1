//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package tensor

import (
	"fmt"
	"io"
	"os"
	"log"
)

// TODO(general): refactor all names to go convention: tensor.Print(), sim.New(), assert.Check(), ...
// use "." imports a bit less.


/** Writes the tensor in 32-bit binary format. See libtensor.h for more details. */
func Write(out io.Writer, t Tensor) {
	//fmt.Fprintln(os.Stderr, "WriteTensor(", &t, out, ")");
	buffer := make([]byte, 4)

	IntToBytes(Rank(t), buffer) // probably could have used unformatted printf, scanf here?
	_, err := out.Write(buffer)
	if err != nil {
		log.Crash(err)
	}

	for i := range t.Size() {
		IntToBytes(t.Size()[i], buffer)
		_, err := out.Write(buffer)
		if err != nil {
			log.Crash(err)
		}
	}

	for i := NewIterator(t); i.HasNext(); i.Next() {
		FloatToBytes(i.Get(), buffer)
		_, err := out.Write(buffer)
		if err != nil {
			log.Crash(err)
		}
	}
}

func WriteFile(fname string, t Tensor) {
	file := FOpenz(fname)
	Write(file, t)
	file.Close()
}

/** Reads a tensor from 32-bit binary format. See libtensor.h for more details. */
func Read(in io.Reader) StoredTensor {
	//fmt.Fprintln(os.Stderr, "ReadTensor(", in, ")");
	buffer := make([]byte, 4)

	in.Read(buffer)
	rank := BytesToInt(buffer)

	size := make([]int, rank)
	for i := range size {
		in.Read(buffer)
		size[i] = BytesToInt(buffer)
	}

	t := NewTensorN(size)
	list := t.List()

	for i := range list {
		in.Read(buffer)
		list[i] = BytesToFloat(buffer)
	}

	return t
}

func ReadFile(file string) StoredTensor {
	return Read(FOpenz(file))
}

/** Prints the tensor in ASCII format, See libtensor.h for more details. */
func Print(out io.Writer, t Tensor) {
	fmt.Fprintln(out, Rank(t))

	for i := range t.Size() {
		fmt.Fprintln(out, t.Size()[i])
	}

	for i := NewIterator(t); i.HasNext(); i.Next() {
		fmt.Fprintln(out, i.Get())
	}
}

/** Prints the tensor in ASCII with some row/column formatting to make it easier to read for humans. */
func Format(out io.Writer, t Tensor) {
	//   fmt.Fprintln(out, t)
	//
	//   if t == nil{                       // seems not to work on interfaces
	//     fmt.Fprintln(out, "(nil)")
	//     return
	//   }

	for i := range t.Size() {
		fmt.Fprint(out, t.Size()[i], " ")
	}
	fmt.Fprintln(out)

	for i := NewIterator(t); i.HasNext(); i.Next() {
		//fmt.Fprintf(out, "%f ", i.Get()); // %15f
		fmt.Fprint(out, i.Get(), "\t")

		for j := 0; j < Rank(t); j++ {
			newline := true
			for k := j; k < Rank(t); k++ {
				if i.Index()[k] != t.Size()[k]-1 {
					newline = false
				}
			}
			if newline {
				fmt.Fprint(out, "\n")
			}
		}

	}
}

/** Prints an unstructured field of vectors (3 co-ordinates and 3 vector components per line), suitable for Gnuplot 'plot with vectors' */
func PrintVectors(out io.Writer, t Tensor) {
	assertMsg(t.Size()[0] == 3, "Needs first dimension of size 3 (vector components)")
	xcomp := Slice(t, 0, 0) //X
	ycomp := Slice(t, 0, 1) //Y
	zcomp := Slice(t, 0, 2) //Z

	for it := NewIterator(xcomp); it.HasNext(); it.Next() {
		index := it.Index()
		for i := 0; i < len(it.Index()); i++ {
			fmt.Fprintf(out, "% d", index[i])
		}
		fmt.Fprint(out, " ", xcomp.Get(index), ycomp.Get(index), zcomp.Get(index), "\n")
	}

	// close if possible.
	c := out.(io.Closer)
	if c != nil {
		out.(io.Closer).Close()
	}
}


/* Todo: sometimes appends instead of overwriting...
move to util?
TODO: remove the duplicate in util
*/

func FOpen(filename string) io.Writer {
	file, ok := os.Open(filename, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
	if ok != nil {
		fmt.Fprint(os.Stderr, ok, "\n")
		log.Crash("Could not open file")
	}
	return file
}
