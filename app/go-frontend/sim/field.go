package sim

import(

)

type Field struct{
  Magnet
  Conv
  // exchange, ...
}



func NewField(b Backend, m Magnet) *Field{
  field := new(Field)
  field.Magnet = m
  //field.Conv = 
  return field
}



