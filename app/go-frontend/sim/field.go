package sim

import(
  "tensor"
)

// Field contains everything neccesary to calculate the effective field H_eff
type Field struct{
  Magnet
  
  *Conv
  // Exchange
  // Anis
  // ...
}

// Takes the parameters from a Magnet struct and
// initializes everything.
func NewField(dev Backend, mag *Magnet) *Field{
  
  field := new(Field)
  
  field.Magnet = *mag

  demag := FaceKernel(field.Size(), field.CellSize())
  exch := Exch6NgbrKernel(field.Size(), field.CellSize())
  kernel := toSymmetric(tensor.Buffer(tensor.Add(demag, exch)))
  field.Conv = NewConv(dev, field.Size(), kernel)
  
  return field
}


// Calculates the effective field of m and stores it in h
func(f *Field) CalcHeff(m, h *Tensor){
  Debugvv("Field.CalcHeff()")
  f.Convolve(m, h)
} 


func(f *Field) String() string{
  return "Field:\n" + f.Magnet.String()
}

