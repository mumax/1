package sim

import(
   "testing"
)

func TestEuler(t *testing.T){
  alpha := 1.0
  dt := 1E-6
  
  field := NewField()
  _ = NewEuler(field, alpha, dt)

  
}
