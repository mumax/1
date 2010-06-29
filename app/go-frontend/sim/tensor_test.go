package sim

import(
  "testing"
  "tensor"
  "fmt"
)


func TestCopy(t *testing.T){
  size := []int{4, 8, 16}
  dev1, dev2 := NewTensor(size), NewTensor(size)
  host1, host2 := tensor.NewTensorN(size), tensor.NewTensorN(size)

  for i:=range host1.List() {
    host1.List()[i] = float(i)
  }

  TensorCopyTo(host1, dev1)
  TensorCopyOn(dev1, dev2)
  TensorCopyFrom(dev2, host2)

  assert(tensor.Equals(host1, host2))
}

func TestGetSet(t *testing.T){
  tens := NewTensor([]int{5, 6, 7})
  fmt.Println(tens)
  tens.Set([]int{4, 5, 6}, 3.14)
  if tens.Get([]int{4, 5, 6}) != 3.14 { t.Fail() }
}
