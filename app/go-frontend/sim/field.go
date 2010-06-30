package sim

import(
  "tensor"
)

type Field struct{
  Magnet
  
  Conv
  // Exchange
  // Anis
  // ...
}



func NewField(dev Backend, mag *Magnet) *Field{
  
  field := new(Field)
  
  field.Magnet = *mag

  demag := FaceKernel(field.Size(), field.CellSize())
  exch := Exch6NgbrKernel(field.Size(), field.CellSize())
  kernel := toSymmetric(tensor.Buffer(tensor.Add(demag, exch)))
  field.Conv = *NewConv(dev, field.Size(), kernel)
  
  return field
}



