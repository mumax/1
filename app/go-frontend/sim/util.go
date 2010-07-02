package sim

import(
  "log"
)


func assert(b bool){
  if !b{
    log.Crash("assertion failed");
  }
}


func Size4D(size3D []int) []int{
  assert(len(size3D) == 3)
  size4D := make([]int, 4)
  size4D[0] = 3
  for i:=range size3D{
    size4D[i+1] = size3D[i]
  }
  return size4D
}

