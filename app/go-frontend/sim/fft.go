package gpu

import(
  "fmt"
  "unsafe"
  "tensor"
)

type Device interface{
  Malloc()
}

/// 3D real-to-complex / complex-to-real transform. Handles zero-padding efficiently (if applicable)
type FFT struct{
  Device
  plan unsafe.Pointer           ///< points to the gpuFFT3dPlan struct that does the actual FFT
  dataSize   [3]int             ///< size of the non-zero data inside the logic input data. Must be <= logicSize
  logicSize  [3]int             ///< logical size of the FFT, including padding: number of reals in each dimension
  physicSize [3]int             ///< The input data needs to be padded with zero's to physicSize, in order to accomodate for the extra complex number in the last dimension needed by real-to-complex FFTS. Additionally, even extra zero's are probably going to be added to fit the gpu stride.
}


/// logicSize is the size of the real input data.
func NewFFT(logicSize []int) *FFT{
  return NewFFTPadded(logicSize, logicSize)
}


/**
 * logicSize is the size of the real input data, but this may contain a lot of zeros.
 * dataSize is the portion of logicSize that is non-zero (typically half as large as logicSize).
 */
func NewFFTPadded(dataSize, logicSize []int) *FFT{
  assert(len(logicSize) == 3)
  assert(len(dataSize) == 3)
  for i:=range dataSize{
    assert(dataSize[i] <= logicSize[i])
  }

  fft := new(FFT)
  for i:=range logicSize {
    fft.logicSize [i] = logicSize[i]
    fft.dataSize  [i] = dataSize[i]
    fft.physicSize[i] = fft.logicSize[i] // Z will be overwritten
  }
  fft.physicSize[Z] = PadToStride(logicSize[Z] + 2)
  fft.plan = NewFFTPlan(dataSize, logicSize)
  
  return fft
}


func (fft *FFT) Forward(in, out *Tensor){
  // size checks
  assert(tensor.Rank(in) == 3)
  assert(tensor.Rank(out) == 3)
  for i,s := range fft.physicSize{
    assert(  in.size[i] == s)
    assert( out.size[i] == s)
  }
  // actual fft
  FFTForward(fft.plan, in, out);
}


func (fft *FFT) Inverse(in, out *Tensor){
  // size checks
  assert(tensor.Rank(in) == 3)
  assert(tensor.Rank(out) == 3)
  for i,s := range fft.physicSize{
    assert(  in.size[i] == s)
    assert( out.size[i] == s)
  }
  // actual fft
  FFTInverse(fft.plan, in, out)
}


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


