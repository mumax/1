package gpu

import(
  "testing"
)

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
