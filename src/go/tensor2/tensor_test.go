package tensor2

import (
	"testing"
	"os"
// 	"fmt"
)

func TestMisc(test *testing.T) {
	t := NewT4([]int{1, 2, 3, 4})
  for i:= range t.TList{
    t.TList[i] = 3.;
  }

	out, err := os.Open("test.tensor", os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	if err != nil {
		panic(err)
	}
	metadata := map[string]string{"author":"arne", "hello":"world!"}
	WriteMetaTensorBinary(out, t, metadata)
	out.Close()

	in, err2 := os.Open("test.tensor", os.O_RDONLY, 0666)
	defer in.Close()
	if err2 != nil {
		panic(err)
	}
  t2 := Read(in)
  WriteAscii(os.Stdout, t2)
}
