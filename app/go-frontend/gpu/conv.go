package gpu

import(

)


type Conv struct{
  fft *FFT;
  kernel [3][3]*Tensor
}


func NewConv(kernel [3][3]*Tensor) *Conv{
  conv := new(Conv)

  return conv
}

