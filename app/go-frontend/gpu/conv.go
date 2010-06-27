package gpu

import(
  "tensor"
)


type Conv struct{
  kernel     [6]*Tensor
  buffer     [3]*Tensor
  mComp      [3]*Tensor
  hComp      [3]*Tensor
  fft       *FFT;
}


func NewConv(dataSize, kernelSize []int) *Conv{
  assert(len(dataSize) == 3)
  assert(len(kernelSize) == 3)
  for i:=range dataSize{
    assert(dataSize[i] <= kernelSize[i])
  }

  conv := new(Conv)
  conv.fft = NewFFTPadded(dataSize, kernelSize)
  
  ///@todo do not allocate for infinite2D problem
  for i:=range conv.buffer{
    conv.buffer[i] = NewTensor(conv.PhysicSize())
//     mComp[i] = &Tensor{ dataSize, }
//     hComp[i] = &Tensor{ dataSize, }
  }
  return conv
}


func (conv *Conv) Exec(source, dest *Tensor){
  assert(len(source.size) == 4)
  assert(len(  dest.size) == 4)
  for i,s:= range conv.DataSize(){
    assert(source.size[i+1] == s)
    assert(  dest.size[i+1] == s)
  }
  
  
//   for i:=0; i<3; i++{
//     CopyPad()
//   }
  
}


func (conv *Conv) LoadKernel6(kernel []*tensor.Tensor3){
  // size checks

  buffer := tensor.NewTensorN(conv.KernelSize())
  devbuf := NewTensor(conv.KernelSize())

  fft := NewFFT(conv.KernelSize())
  for i:= range conv.kernel{
    if kernel[i] != nil{
      conv.kernel[i] = NewTensor(conv.PhysicSize())

      tensor.CopyTo(kernel[i], buffer)
      TensorCopyTo(buffer, devbuf)
      CopyPad(devbuf, conv.kernel[i])   ///@todo padding should be done on host, not device, to save gpu memory / avoid fragmentation

      fft.Forward(conv.kernel[i], conv.kernel[i])
    }
  }
}



/// size of the magnetization and field, this is the FFT dataSize
func (conv *Conv) DataSize() []int{
  return conv.fft.DataSize()
}


/// size of magnetization + padding zeros, this is the FFT logicSize
func (conv *Conv) KernelSize() []int{
  return conv.fft.LogicSize()
}


/// size of magnetization + padding zeros + striding zeros, this is the FFT logicSize
func (conv *Conv) PhysicSize() []int{
  return conv.fft.PhysicSize()
}

const(
  XX = 0
  YY = 1
  ZZ = 2
  YZ = 3
  XZ = 4
  XY = 5
)



