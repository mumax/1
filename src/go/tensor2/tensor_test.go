package tensor2

import (
	"testing"
	"os"
// 	"fmt"
)

func TestMisc(test *testing.T) {
	t := NewT4([]int{1, 2, 3, 4})

	out, err := os.Open("test.tensor", os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	if err != nil {
		panic(err)
	}
	Write(out, t)
	out.Close()

	in, err2 := os.Open("test.tensor", os.O_RDONLY, 0666)
	defer in.Close()
	if err2 != nil {
		panic(err)
	}
  t2 := Read(in)
  WriteAscii(os.Stdout, t2)
}
