package gpu

import(
   "testing"
   "tensor"
    "os"
    "fmt"
    "sim"
)

func TestConv(t *testing.T){
  size := []int{4, 8, 16}
  kernelSize := []int{8, 16, 32}
  
  conv := NewConv(size, kernelSize)
  conv.LoadKernel6(sim.FaceKernel6(size))
  
  for i:=range(conv.kernel){
    fmt.Println(i)
    tensor.Format(os.Stdout, conv.kernel[i])
  }
}
