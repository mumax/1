package omf

import (
	"testing"
	"tensor"
	"iotool"
	"fmt"
)

type Test struct{}

func (t *Test) GetData() (data tensor.Interface, multiplier float32, unit string) {
	tens := tensor.NewT4([]int{3, 128, 32, 1})
	for i:= range tens.List(){
      tens.List()[i] = float32(i+1)
    }
    data, multiplier, unit = tens, 800e3, "A/m"
	return
}

func (t *Test) GetMetadata() map[string]string {
	return map[string]string{"key1": "val1", "key2": "val2"}
}

func (t *Test) GetMesh() (cellsize []float32, unit string) {
	cellsize = []float32{1e-9, 1e-9, 50e-9}
	unit = "m"
	return
}


func TestIO(test *testing.T) {
	t := &Test{}
	orig, _, _ := t.GetData()
	codec := NewOmfCodec()
	codec.Encode(iotool.MustOpenWRONLY("test.omf"), t)
	tens, _ := codec.Decode(iotool.MustOpenRDONLY("test.omf"))
	fmt.Println(tens.Size())
	if !tensor.Equal(orig, tens) {test.Fail()}
}
