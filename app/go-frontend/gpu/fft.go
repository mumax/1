package gpu

import(
  "fmt"
)


/**
 * The physical size (needed for storage) corresponding to this
 * FFT's logical size. It is at least 2 floats larger in the Z dimension,
 * and usually even more due to GPU striding.
 */
func (fft *FFT) PhysicSize() []int{
  return fft.physicSize[0:]
}


/**
 * Size of the actual data being transformed.
 * This may contain a lot of padding zeros that
 * are handled efficiently.
 */
func (fft *FFT) LogicSize() []int{
  return fft.logicSize[0:]
}

/**
 * Portion of the logical size that is nonzero
 */
func (fft *FFT) DataSize() []int{
  return fft.dataSize[0:]
}


func (fft *FFT) String() string{
  return fmt.Sprint("FFT{ dataSize", fft.dataSize, "logicSize", fft.logicSize, "physicSize", fft.physicSize, "}");
}


