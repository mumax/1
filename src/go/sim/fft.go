package sim

import (
	"fmt"
	"unsafe"
	"tensor"
)


// 3D real-to-complex / complex-to-real transform. Handles zero-padding efficiently (if applicable)
type FFT struct {
	*Backend
	plan       unsafe.Pointer ///< points to the simFFT3dPlan struct that does the actual FFT
	dataSize   [3]int         ///< size of the non-zero data inside the logic input data. Must be <= logicSize
	logicSize  [3]int         ///< logical size of the FFT, including padding: number of reals in each dimension
	physicSize [3]int         ///< The input data needs to be padded with zero's to physicSize, in order to accomodate for the extra complex number in the last dimension needed by real-to-complex FFTS. Additionally, even extra zero's are probably going to be added to fit the sim stride.
}


// logicSize is the size of the real input data.
func NewFFT(b *Backend, logicSize []int) *FFT {
	return NewFFTPadded(b, logicSize, logicSize)
}


/**
 * logicSize is the size of the real input data, but this may contain a lot of zeros.
 * dataSize is the portion of logicSize that is non-zero (typically half as large as logicSize).
 */
func NewFFTPadded(b *Backend, dataSize, logicSize []int) *FFT {
	assert(len(logicSize) == 3)
	assert(len(dataSize) == 3)
	for i := range dataSize {
		assert(dataSize[i] <= logicSize[i])
	}

	fft := new(FFT)
	fft.Backend = b
	for i := range logicSize {
		fft.logicSize[i] = logicSize[i]
		fft.dataSize[i] = dataSize[i]
		fft.physicSize[i] = fft.logicSize[i] // Z will be overwritten
	}
	fft.physicSize[Z] = logicSize[Z] + 2
	fft.plan = fft.newFFTPlan(dataSize, logicSize)

	return fft
}


func (fft *FFT) Forward(in, out *Tensor) {
	// size checks
	assert(tensor.Rank(in) == 3)
	assert(tensor.Rank(out) == 3)
	for i, s := range fft.dataSize {
    assert(in.size[i] == s)
  }
  for i, s := range fft.physicSize {
    assert(out.size[i] == s)
  }
	// actual fft
  fft.Start("FFT forward")
	fft.fftForward(fft.plan, in.data, out.data)
	fft.Stop("FFT forward")
}



func (fft *FFT) Inverse(in, out *Tensor) {
	// size checks
	assert(tensor.Rank(in) == 3)
	assert(tensor.Rank(out) == 3)
  for i, s := range fft.physicSize {
    assert(in.size[i] == s)
  }
  for i, s := range fft.dataSize {
    assert(out.size[i] == s)
  }
	// actual fft
  fft.Start("FFT inverse")
	fft.fftInverse(fft.plan, in.data, out.data)
	fft.Stop("FFT inverse")
}


func (fft *FFT) InverseInplace(data *Tensor) {
	fft.Inverse(data, data)
}


/**
 * The physical size (needed for storage) corresponding to this
 * FFT's logical size. It is at least 2 floats larger in the Z dimension,
 * and usually even more due to GPU striding.
 */
func (fft *FFT) PhysicSize() []int {
	return fft.physicSize[0:]
}


/**
 * Size of the actual data being transformed.
 * This may contain a lot of padding zeros that
 * are handled efficiently.
 */
func (fft *FFT) LogicSize() []int {
	return fft.logicSize[0:]
}

/**
 * Portion of the logical size that is nonzero
 */
func (fft *FFT) DataSize() []int {
	return fft.dataSize[0:]
}


func (fft *FFT) Normalization() int {
	return fft.logicSize[X] * fft.logicSize[Y] * fft.logicSize[Z]
}

func (fft *FFT) String() string {
	return fmt.Sprint("FFT{ dataSize", fft.dataSize, "logicSize", fft.logicSize, "physicSize", fft.physicSize, "}")
}
