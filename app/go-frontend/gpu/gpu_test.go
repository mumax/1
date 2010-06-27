package gpu

import(
  "testing"
  "tensor"
)

func TestPad(t *testing.T){
  small := []int{4, 8, 16}
  big := []int{6, 12, 32}
  
  dev1, dev2 := NewTensor(small), NewTensor(big)
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

func TestTensor(t *testing.T){
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



func abs(r float) float{
  if r < 0 { return -r }
  //else
  return r
}

func TestStride(t *testing.T){
  s := Stride()
  OverrideStride(10)
  if Stride() != 10 { t.Fail() }

  for i:=1; i<100; i++{
    if PadToStride(i) % Stride() != 0 { t.Fail() }
  }
  
  OverrideStride(-1)
  if Stride() != s { t.Fail() }
}


func TestMisc(t *testing.T){
  PrintProperties()
}


func TestZero(t *testing.T){
  N := 100
  host := make([]float, N)
  dev := NewArray(N)

  for i:=range(host){
    host[i] = float(i);
  }

  MemcpyTo(&host[0], dev, N)
  Zero(dev, N/2)
  MemcpyFrom(dev, &host[0], N)

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
  dev1, dev2 := NewArray(N), NewArray(N)

  for i:=range(host1){
    host1[i] = float(i);
  }
  
  MemcpyTo(&host1[0], dev1, N)
  MemcpyOn(dev1, dev2, N)
  MemcpyFrom(dev2, &host2[0], N)

  for i:=range(host1){
    if host1[i] != host2[i] { t.Fail() }
  }
}
