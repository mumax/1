package omf

import (
  "testing"
  "tensor"
  "os"
)

func TestIO(test *testing.T){
  size := []int{3, 1, 128, 32}
  t := tensor.NewT4(size)
  meta := make(map[string]string)
  codec := NewOmfCodec()
  codec.Encode(os.Stdout, t, meta)

}