package sim

import(
   "testing"
)

func TestEuler(t *testing.T){
  alpha := 1
  dt := 1E-6
  
  field := NewField()
  solver := NewEuler(field, alpha, dt)

}
