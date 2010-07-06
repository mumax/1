package main

import(
  . "sim"
  "tensor"
  "fmt"
)

func main(){
  Verbosity = 2
  
  dev := GPU

  mat := NewMaterial()
  mat.MSat = 800E3
  mat.AExch = 1.3E-11
  mat.Alpha = 1.0
  
  size := []int{1, 32, 128}
  L := mat.UnitLength()
  cellsize := []float{3E-9 / L, 3.90625E-9 / L, 3.90625E-9 / L}

  magnet := NewMagnet(dev, mat, size, cellsize)
  
  dt := 0.1E-12 / mat.UnitTime()
  solver := NewEuler(dev, magnet, dt)
 
  
  fmt.Println(solver)

  m := tensor.NewTensorN(Size4D(magnet.Size()))
  for i:=range m.List(){
    m.List()[i] = 1.
  }
  TensorCopyTo(m, solver.M())

  file:=0
  for i:=0; i<100; i++{
    TensorCopyFrom(solver.M(), m)
    fname := "m" + fmt.Sprintf("%06d", file) + ".t"
    file++
    tensor.WriteFile(fname, m)
    for j:=0; j<100; j++{
      solver.Step()
    }
  }

  solver.Dt = 0.01E-12 / mat.UnitTime()
  solver.Alpha = 0.02
  B := solver.UnitField()
  solver.Hext = []float{0/B, 4.3E-3/B, -24.6E-3/B}
  
  for i:=0; i<1000; i++{
    TensorCopyFrom(solver.M(), m)
    fname := "m" + fmt.Sprintf("%06d", file) + ".t"
    file++
    tensor.WriteFile(fname, m)
    for j:=0; j<1000; j++{
      solver.Step()
    }
  }
  
  TimerPrintDetail()
}



