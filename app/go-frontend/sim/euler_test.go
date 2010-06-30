package sim

import(
   "testing"
)

func TestEuler(t *testing.T){
  alpha := 1.0
  dt := 1E-6

  size := []int{1, 32, 128}
  cellsize := []float{1., 1., 1.}
  
  dev := GPU
  mat := *NewMaterial()
  mat.MSat = 800E3
  mat.AExch = 1.1E-13
  mag := *NewMagnet(mat, size, cellsize)
  field := NewField(dev, mag) // to be constructed by solver
  
  _ = NewEuler(field, alpha, dt)

  
}
