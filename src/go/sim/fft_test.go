package sim

import(
  "testing"
  "tensor"
//    "os"
   "fmt"
)


func TestFFTOutOfPlace(t *testing.T){

  //backend.OverrideStride(1)

  sizes := [][]int{
    []int{1, 64, 64},
    []int{2, 64, 64}}

    for _, size := range sizes{
      fft := NewFFT(backend, size)
      fmt.Println(fft)
      physicSize := fft.PhysicSize()

      devLog, devPhys1, devPhys2 := NewTensor(backend, size), NewTensor(backend, physicSize), NewTensor(backend, physicSize)
      host1, host2 := tensor.NewTensorN(size), tensor.NewTensorN(size)

      for i:=0; i<tensor.N(host1); i++ {
        host1.List()[i] = float(i % 100) / 100
      }

      TensorCopyTo(host1, devLog)
      CopyPad(devLog, devPhys1)
      //tensor.Format(os.Stdout, devPhys)
      fft.Forward(devPhys1, devPhys2)
      //tensor.Format(os.Stdout, devPhys)
      ZeroTensor(devPhys1)
      fft.Inverse(devPhys2, devPhys1)
      //tensor.Format(os.Stdout, devPhys)
      CopyUnpad(devPhys1, devLog)
      //tensor.Format(os.Stdout, devLog)
      TensorCopyFrom(devLog, host2)

      N := float(fft.Normalization());
      var maxError float = 0
      for i:=range host2.List() {
        host2.List()[i] /= N;
        if abs(host2.List()[i] - host1.List()[i]) > maxError {
          maxError = abs(host2.List()[i] - host1.List()[i])
        }
      }
      //tensor.Format(os.Stdout, host2)
      fmt.Println("FFT error:", maxError);
      if maxError > 1E-4 { t.Fail() }
    }
    //backend.OverrideStride(-1)
}


func TestFFTInplace(t *testing.T){

  //backend.OverrideStride(1)

  sizes := [][]int{
    []int{1, 64, 64},
    []int{2, 64, 64}}

    for _, size := range sizes{
      fft := NewFFT(backend, size)
      fmt.Println(fft)
      physicSize := fft.PhysicSize()

      devLog, devPhys := NewTensor(backend, size), NewTensor(backend, physicSize)
      host1, host2 := tensor.NewTensorN(size), tensor.NewTensorN(size)

      for i:=0; i<tensor.N(host1); i++ {
        host1.List()[i] = float(i % 100) / 100
      }

      TensorCopyTo(host1, devLog)
      CopyPad(devLog, devPhys)
      //tensor.Format(os.Stdout, devPhys)
      fft.Forward(devPhys, devPhys)
      //tensor.Format(os.Stdout, devPhys)
      fft.Inverse(devPhys, devPhys)
      //tensor.Format(os.Stdout, devPhys)
      CopyUnpad(devPhys, devLog)
      //tensor.Format(os.Stdout, devLog)
      TensorCopyFrom(devLog, host2)

      N := float(fft.Normalization());
      var maxError float = 0
      for i:=range host2.List() {
        host2.List()[i] /= N;
        if abs(host2.List()[i] - host1.List()[i]) > maxError {
          maxError = abs(host2.List()[i] - host1.List()[i])
        }
      }
      //tensor.Format(os.Stdout, host2)
      fmt.Println("FFT error:", maxError);
      if maxError > 1E-4 { t.Fail() }
    }
    //backend.OverrideStride(-1)
}

func abs(r float) float{
  if r < 0 { return -r }
  //else
  return r
}

