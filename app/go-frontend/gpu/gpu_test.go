package gpu

import(
  "testing"
)

func TestStride(t *testing.T){
  s := Stride();
  OverrideStride(10);
  if Stride() != 10 { t.Fail() }

  for i:=1; i<100; i++{
    if PadToStride(i) % Stride() != 0 { t.Fail() }
  }
  
  OverrideStride(-1);
  if Stride() != s { t.Fail() }
}


func TestMisc(t *testing.T){
  PrintProperties()
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
