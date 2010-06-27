package gpu

import(
  "testing"
  "tensor"
   "os"
   "fmt"
)


func TestFFT(t *testing.T){

  OverrideStride(1)

  size := []int{4, 8, 4}
  fft := NewFFT(size)
  fmt.Println(fft)
  physicSize := fft.PhysicSize()

  devLog, devPhys := NewTensor(size), NewTensor(physicSize)
  host1, host2 := tensor.NewTensorN(size), tensor.NewTensorN(size)

  for i:=0; i<4; i++ {
    host1.List()[i] = float(i)
  }

  TensorCopyTo(host1, devLog)
  CopyPad(devLog, devPhys)
  tensor.Format(os.Stdout, devPhys)
  fft.Forward(devPhys, devPhys)
  tensor.Format(os.Stdout, devPhys)
  fft.Inverse(devPhys, devPhys)
  tensor.Format(os.Stdout, devPhys)
  CopyUnpad(devPhys, devLog)
  tensor.Format(os.Stdout, devLog)
  TensorCopyFrom(devLog, host2)

  N := float(fft.Normalization());
  var maxError float = 0
  for i:=range host2.List() {
    host2.List()[i] /= N;
    if abs(host2.List()[i] - host1.List()[i]) > maxError {
      maxError = abs(host2.List()[i] - host1.List()[i])
    }
  }
  tensor.Format(os.Stdout, host2)
  fmt.Println("FFT error:", maxError);
  if maxError > 1E-5 { t.Fail() }

  OverrideStride(-1)
}
