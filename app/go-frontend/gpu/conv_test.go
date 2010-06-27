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
  kernel := sim.FaceKernel6(size, []float{1., 1., 1.})
  
  for i,k:= range kernel{
    fmt.Println("kernel", i, k.Size())
  }
  fmt.Println("conv", conv.KernelSize())
  
  conv.LoadKernel6(kernel)
  
  for i:=range(conv.kernel){
    fmt.Println(i)
    if conv.kernel[i] == nil {
      fmt.Println("(nil)")
    }else{
      tensor.Format(os.Stdout, conv.kernel[i])
    }
  }
}
