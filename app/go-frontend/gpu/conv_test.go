package gpu

import(
   "testing"
 //   "sim"
    "tensor"
    "fmt"
    "os"
)

func TestConv(t *testing.T){
  OverrideStride(1)
  size4D := []int{3, 4, 4, 2}
  size := size4D[1:]
  kernelSize := []int{2*size[X], 2*size[Y], 2*size[Z]}
  
  conv := NewConv(size, kernelSize)
  //kernel := sim.FaceKernel6(size, []float{1., 1., 1.})
  kernel := make([]*tensor.Tensor3, 6)
  for i := range kernel{
    kernel[i] = tensor.NewTensor3(kernelSize)
  }
  kernel[XX].List()[0] = 1.
  
//   for i,k:= range kernel{
//     fmt.Println("kernel", i, k.Size())
//   }
//   fmt.Println("conv", conv.KernelSize())
//   fmt.Println("conv.fft", conv.fft)
  
  conv.LoadKernel6(kernel)
  
  for i:=range(conv.kernel){
    fmt.Println(i)
    if conv.kernel[i] == nil {
      fmt.Println("(nil)")
    }else{
      tensor.Format(os.Stdout, conv.kernel[i])
    }
  }

  m, h := NewTensor(size4D), NewTensor(size4D)

  m.Set([]int{0, 0, 0, 0}, 1.)
  tensor.WriteFile("m.t", m)
  conv.Exec(m, h)
  tensor.WriteFile("h.t", h)
  OverrideStride(-1)
}
