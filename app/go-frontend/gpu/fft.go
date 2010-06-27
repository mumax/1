package gpu

import(
 
)


/**
 * Returns the physical size (needed for storage) corresponding to this
 * FFT's logical size. It is at least 2 floats larger in the Z dimension,
 * and usually even more due to GPU striding.
 */
func (fft *FFT) PhysicSize() []int{
  return fft.physicSize[0:]
}

func (fft *FFT) LogicSize() []int{
  return fft.logicSize[0:]
}

func (fft *FFT) DataSize() []int{
  return fft.dataSize[0:]
}





