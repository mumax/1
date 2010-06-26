package gpu

import(
   "testing"
//   "tensor"
//    "os"
//    "fmt"
)

func TestConv(t *testing.T){
  size := []int{4, 8, 16}
  kernelSize := []int{8, 16, 32}
  
  NewConv(size, kernelSize)
}
