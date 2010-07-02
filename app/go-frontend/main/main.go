package main

import(
  . "sim"
  "tensor"
  "strconv"
)

func main(){

  dev := GPU

  mat := NewMaterial()
  mat.MSat = 800E3
  mat.AExch = 1.1E-13
  mat.Alpha = 1.0
  
  size := []int{1, 32, 128}
  L := mat.UnitLength()
  cellsize := []float{3E-9 / L, 3E-9 / L, 3E-9 / L}

  magnet := NewMagnet(mat, size, cellsize)
  
  dt := 0.1E-15 / mat.UnitTime()
  
  solver := NewEuler(dev, magnet, dt)

  m := tensor.NewTensorN(Size4D(magnet.Size()))
  for i:=range m.List(){
    m.List()[i] = 1.
  }
  TensorCopyTo(m, solver.M())
  
  for i:=0; i<100; i++{
    TensorCopyFrom(solver.M(), m)
    fname := "m" + strconv.Itoa(i) + ".t"
    tensor.WriteFile(fname, m)
    for j:=0; j<100; j++{
      solver.Step()
    }
  }
  
  TimerPrintDetail()
}



