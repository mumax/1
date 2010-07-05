package main

import(
  . "sim"
  "tensor"
  "fmt"
)

func main(){
  Verbosity = 1
  
  dev := GPU

  mat := NewMaterial()
  mat.MSat = 800E3
  mat.AExch = 1.3E-11
  mat.Alpha = 1.0
  
  size := []int{1, 32, 128}
  L := mat.UnitLength()
  cellsize := []float{3E-9 / L, 3.90625E-9 / L, 3.90625E-9 / L}

  magnet := NewMagnet(mat, size, cellsize)
  
  dt := 0.1E-12 / mat.UnitTime()
  
  solver := NewEuler(dev, magnet, dt)

  fmt.Println(solver)

  m := tensor.NewTensorN(Size4D(magnet.Size()))
  for i:=range m.List(){
    m.List()[i] = 1.
  }
  TensorCopyTo(m, solver.M())
  
  for i:=0; i<100; i++{
    TensorCopyFrom(solver.M(), m)
    fname := "m" + fmt.Sprintf("%06d", i) + ".t"
    tensor.WriteFile(fname, m)
    for j:=0; j<100; j++{
      solver.Step()
    }
  }
  
  TimerPrintDetail()
}



