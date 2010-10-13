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
)

const(
  H_COMMENT = "#"
  H_SEPARATOR = ":"
  H_FORMAT = "tensorformat"
)

func WriteHeader(out_ io.Writer, t Interface){
  out := bufio.NewWriter(out_)
  defer out.Flush()

  fmt.Fprintln(out, H_COMMENT, H_FORMAT, H_SEPARATOR, 1)
}

func WriteDataAscii(out_ io.Writer, t Interface) {
  out := bufio.NewWriter(out_)
  defer out.Flush()
  
  for i := NewIterator(t); i.HasNext(); i.Next() {
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
