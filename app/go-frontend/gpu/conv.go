package gpu

import(
  "tensor"
)


type Conv struct{
//   dataSize   [3]int                 ///< size of the magnetization and field, this becomes the FFT dataSize
//   logicSize  [3]int                 ///< size of magnetization + padding zeros, this becomes the FFT logicSize
//   physicSize [3]int                 ///< logic size + even more padding zeros

  kernel     [6]*Tensor                 ///< should have logicSize
  fft       *FFT;
}

const(
  XX = 0
  YY = 1
  ZZ = 2
  YZ = 3
  XZ = 4
  XY = 5
)

func NewConv(dataSize, kernelSize []int) *Conv{
  ///@todo size check
  
  conv := new(Conv)
  conv.fft = NewFFTPadded(dataSize, kernelSize)
  return conv
}

func (conv *Conv) DataSize() []int{
  return conv.fft.DataSize()
}

func (conv *Conv) KernelSize() []int{
  return conv.fft.LogicSize()
}

func (conv *Conv) LoadKernel9(kernel tensor.Tensor){
//   fft := NewFFT(conv.KernelSize())
//   for i:= range conv.kernel{
//     conv.kernel[i] = NewArray(conv.fft.physicSize);
//   }
}

