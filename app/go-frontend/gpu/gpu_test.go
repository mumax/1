package gpu

import(
  "testing"
  "tensor"
)

func TestTensor(t *testing.T){
  
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
