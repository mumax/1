package main

import(
  . "sim"
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

  for i:=0; i<1000; i++{
    solver.Step()
  }
  TimerPrintDetail()
}



