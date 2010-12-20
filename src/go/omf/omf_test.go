package omf

import (
	"testing"
	"tensor"
	"os"
)

type Test struct{}

func (t *Test)  GetData() (data tensor.Interface, multiplier float32, unit string) {
  data, multiplier, unit = tensor.NewT4([]int{3, 128, 32, 1}), 800e3, "A/m"
  return
}

func (t *Test)    GetMetadata() map[string]string{
  return map[string]string{"key1":"val1", "key2":"val2"}
}

func (t *Test)    GetMesh() (cellsize []float32, unit string){
  cellsize =  []float32{1e-9, 1e-9, 50e-9}
  unit = "m"
  return
}


func TestIO(test *testing.T) {
	t := &Test{}
	codec := NewOmfCodec()
	codec.Encode(os.Stdout, t)
}
