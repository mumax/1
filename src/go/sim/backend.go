package sim

import (
	"tensor"
	"unsafe"
)

/**
 * A Backend wraps some unsafe methods of the Device interface
 * with safe versions, and provides additional methods derived
 * from low-end Devices methods. Therefore, the user should
 * interact with a Backend and forget about the underlying Device.
 *
 * Other unsafe methods are wrapped by higher-lever structs that
 * embed a Backend. E.g. FFT wraps the fft functions, Conv wraps
 * convolution functions, etc.
 *
 */

type Backend struct {
	Device
	Initiated bool
	Timer
}

func NewBackend(d Device) *Backend {
	return &Backend{d, false, NewTimer()}
}

//_________________________________________________________________________ safe wrappers for Device methods

// more or less safe initialization, calls the underlying init() only once
// (given you allocate only one unique CPU, GPU, ...)
func (dev *Backend) Init() {
	if !dev.Initiated {
		dev.init()
		dev.Initiated = true
	}
}

// Copies a number of floats from host to GPU
func (dev *Backend) memcpyTo(source *float, dest unsafe.Pointer, nFloats int) {
	dev.memcpy(unsafe.Pointer(source), dest, nFloats, CPY_TO)
}

// Copies a number of floats from GPU to host
func (dev *Backend) memcpyFrom(source unsafe.Pointer, dest *float, nFloats int) {
	dev.memcpy(source, unsafe.Pointer(dest), nFloats, CPY_FROM)
}

// Copies a number of floats from GPU to GPU
func (dev *Backend) memcpyOn(source, dest unsafe.Pointer, nFloats int) {
	dev.memcpy(unsafe.Pointer(source), unsafe.Pointer(dest), nFloats, CPY_ON)
}

// Copies from a smaller to a larger tensor, not touching the additional space in the destination (typically filled with zero padding)
func (dev *Backend) copyPad(source, dest unsafe.Pointer, sourceSize, destSize []int) {
	dev.copyPadded(source, dest, sourceSize, destSize, CPY_PAD)
}

//Copies from a larger to a smaller tensor, not reading the additional data in the source (typically filled with zero padding or spoiled data)
func (dev *Backend) copyUnpad(source, dest unsafe.Pointer, sourceSize, destSize []int) {
	dev.copyPadded(source, dest, sourceSize, destSize, CPY_UNPAD)
}

// Gets one float from a Device array.
// Slow, for debug only
func (dev *Backend) arrayGet(array unsafe.Pointer, index int) float {
	var f float
	dev.memcpyFrom(dev.arrayOffset(array, index), &f, 1)
	return f
}

// Sets one float on a Device array.
// Slow, for debug only
func (dev *Backend) arraySet(array unsafe.Pointer, index int, value float) {
	dev.memcpyTo(&value, dev.arrayOffset(array, index), 1)
}


// adds b to a
func (dev *Backend) Add(a, b *DevTensor) {
	assert(tensor.EqualSize(a.size, b.size))
	dev.add(a.data, b.data, tensor.N(a))
}


// overwrites a with weightA * a + weightB * b
func (dev *Backend) LinearCombination(a, b *DevTensor, weightA, weightB float) {
	assert(tensor.EqualSize(a.size, b.size))
	dev.linearCombination(a.data, b.data, weightA, weightB, tensor.N(a))
}

// adds the constant cnst to each element of a. N = length of a
func (dev *Backend) AddConstant(a *DevTensor, cnst float) {
	Debugvv("Backend.AddConstant(", a, cnst, ")")
	dev.addConstant(a.data, cnst, tensor.N(a))
}

// // adds the constant vector cnst to each element a. len(cnst) == Size(a)[0]
// func(dev *Backend) AddVector(a *Tensor, cnst []float){
//   assert(len(cnst) == a.size[0])
//   for i:=range a.size{
//     dev.addConstant(
//   }
// }

func (dev *Backend) Normalize(m *DevTensor) {
	//Debugvv( "Backend.Normalize()" )
	assert(len(m.size) == 4)
	N := m.size[1] * m.size[2] * m.size[3]
	dev.normalize(m.data, N)
}


// calculates torque * dt, overwrites h with the result
func (dev *Backend) DeltaM(m, h *DevTensor, alpha, dtGilbert float) {
	assert(len(m.size) == 4)
	assert(tensor.EqualSize(m.size, h.size))
	N := m.size[1] * m.size[2] * m.size[3]
	dev.deltaM(m.data, h.data, alpha, dtGilbert, N)
}


func (b Backend) OverrideStride(stride int) {
	panic("OverrideStride is currently not compatible with the used FFT, it should always be 1")
	Debugv("Backend.OverrideStride(", stride, ")")
	assert(stride > 0 || stride == -1)
	b.overrideStride(stride)
}

// unsafe FFT
func (b Backend) fftForward(plan unsafe.Pointer, in, out unsafe.Pointer) {
	b.fft(plan, in, out, FFT_FORWARD)
}

// unsafe FFT
func (b Backend) fftInverse(plan unsafe.Pointer, in, out unsafe.Pointer) {
	b.fft(plan, in, out, FFT_INVERSE)
}

// func (b Backend) ExtractReal(complex, real *Tensor) {
// 	assert(Len(complex.size) == 2*Len(real.size))
// 	b.extractReal(complex.data, real.data, Len(real.size))
// }

//________________________________________________________________________ derived methods


// Takes an array size and returns the smallest multiple of Stride() where the array size fits in
func (b Backend) PadToStride(nFloats int) int {
	stride := b.Stride()
	gpulen := ((nFloats-1)/stride + 1) * stride

	assert(gpulen%stride == 0)
	assert(gpulen > 0)
	assert(gpulen >= nFloats)
	return gpulen
}
