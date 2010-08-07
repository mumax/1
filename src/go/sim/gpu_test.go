package sim

import(
  "testing"
  "tensor"
)

func TestPad(t *testing.T){
  small := []int{4, 8, 16}
  big := []int{6, 12, 32}
  
  dev1, dev2 := NewTensor(CPU, small), NewTensor(CPU, big)
  host1, host2 := tensor.NewTensorN(small), tensor.NewTensorN(small)

  for i:=range host1.List() {
    host1.List()[i] = float(i)
  }

  TensorCopyTo(host1, dev1)
  CopyPad(dev1, dev2)
  ZeroTensor(dev1)
  CopyUnpad(dev2, dev1)
  TensorCopyFrom(dev1, host2)

  assert(tensor.Equals(host1, host2))
}


func TestStride(t *testing.T){
  s := CPU.Stride()
  CPU.OverrideStride(10)
  if CPU.Stride() != 10 { t.Fail() }

  for i:=1; i<100; i++{
    if CPU.PadToStride(i) % CPU.Stride() != 0 { t.Fail() }
  }
  
  CPU.OverrideStride(-1)
  if CPU.Stride() != s { t.Fail() }
}


func TestMisc(t *testing.T){
  CPU.PrintProperties()
}


func TestZero(t *testing.T){
  N := 100
  host := make([]float, N)
  dev := CPU.newArray(N)

  for i:=range(host){
    host[i] = float(i);
  }

  CPU.memcpyTo(&host[0], dev, N)
  CPU.zero(dev, N/2)
  CPU.memcpyFrom(dev, &host[0], N)

  for i:=0; i<N/2; i++{
    if host[i] != 0. { t.Fail() }
  }
  for i:=N/2; i<N; i++{
    if host[i] != float(i) { t.Fail() }
  }
}


func TestMemory(t *testing.T){
  N := 100
  host1, host2 := make([]float, N), make([]float, N)
  dev1, dev2 := CPU.newArray(N), CPU.newArray(N)

  for i:=range(host1){
    host1[i] = float(i);
  }
  
  CPU.memcpyTo(&host1[0], dev1, N)
  CPU.memcpyOn(dev1, dev2, N)
  CPU.memcpyFrom(dev2, &host2[0], N)

  for i:=range(host1){
    if host1[i] != host2[i] { t.Fail() }
  }
}
