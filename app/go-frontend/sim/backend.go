package sim

import(

)


type Backend struct{
  Device
}


//_________________________________________________________________________ safe wrappers for Device methods

func(b Backend) OverrideStride(stride int){
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
