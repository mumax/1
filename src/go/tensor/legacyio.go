package tensor2

// TO BE OBSOLETED

import (
	"io"
	"bufio"
	"fmt"
	"os"
)
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
