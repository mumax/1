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


// TEMP HACK: RANK IS NOT STORED IN ASCII FORMAT
// ASSUME 4
func WriteAscii(out_ io.Writer, t *T) {

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
