package sim

import(
  "tensor"
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

type Backend struct{
  Device
}


//_________________________________________________________________________ safe wrappers for Device methods

// adds b to a
func(dev Backend) Add(a, b *Tensor){
  assert(tensor.EqualSize(a.size, b.size))
  dev.add(a.data, b.data, tensor.N(a))
}


// overwrites a with weightA * a + weightB * b
func(dev Backend) LinearCombination(a, b *Tensor, weightA, weightB float){
  assert(tensor.EqualSize(a.size, b.size))
  dev.linearCombination(a.data, b.data, weightA, weightB, tensor.N(a))
}


func(dev Backend) Normalize(m *Tensor){
  //Debugvv( "Backend.Normalize()" )
  assert(len(m.size) == 4)
  N := m.size[1] * m.size[2] * m.size[3]
  dev.normalize(m.data, N)
}


// calculates torque * dt, overwrites h with the result
func(dev Backend) DeltaM(m, h *Tensor, alpha, dtGilbert float){
  assert(len(m.size) == 4)
  assert(tensor.EqualSize(m.size, h.size))
  N := m.size[1] * m.size[2] * m.size[3]
  dev.deltaM(m.data, h.data, alpha, dtGilbert, N)
}


func(b Backend) OverrideStride(stride int){
  Debugv( "Backend.OverrideStride(", stride, ")" )
  assert(stride > 0 || stride == -1)
  b.overrideStride(stride)
}


//________________________________________________________________________ derived methods


// Takes an array size and returns the smallest multiple of Stride() where the array size fits in
func(b Backend)  PadToStride(nFloats int) int{
  stride := b.Stride()
  gpulen := ((nFloats-1)/stride + 1) * stride;

  assert(gpulen % stride == 0)
  assert(gpulen > 0)
  assert(gpulen >= nFloats)
  return gpulen
}

