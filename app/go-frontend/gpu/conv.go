package gpu

import(

)


type Conv struct{
//   dataSize   [3]int                 ///< size of the magnetization and field, this becomes the FFT dataSize
//   logicSize  [3]int                 ///< size of magnetization + padding zeros, this becomes the FFT logicSize
//   physicSize [3]int                 ///< logic size + even more padding zeros 
  kernel     [3][3]*Tensor             ///< should have paddedSize
  fft       *FFT;
}


func NewConv(dataSize []int, kernel [3][3]*Tensor) *Conv{
  ///@todo size check
  
  conv := new(Conv)

  for i:=range dataSize { conv.dataSize[i] = dataSize[i] }
  
  return conv
}

