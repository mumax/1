package sim

/*
#include "../../../core/tensor.h"
#include "../../../core/gputil.h"
#include "../../../core/gpufft2.h"
#include "../../../core/gpupad.h"
#include "../../../core/gpuconv2.h"
#include "../../../core/gputorque.h"
#include "../../../core/gpueuler.h"
#include "../../../core/gpu_anal.h"
#include "../../../core/gpunormalize.h"
#include "../../../core/timer.h"

// to allow some (evil but neccesary) pointer arithmetic in go
float* gpu_array_offset(float* array, int index){
    return &array[index];
}
*/
import "C"
import "unsafe"

/**
 * This single file intefaces all the relevant CUDA func(d Gpu) tions with go
 * It only wraps the func(d Gpu) tions, higher level constructs and assetions
 * are in separate files like fft.go, ...
 *
 * @note cgo does not seem to like many cgofiles, so I put everything together here.
 * @author Arne Vansteenkiste
 */

import(

)

var GPU Backend = Backend{Gpu{}}

type Gpu struct{
  // intentionally empty, but the methods implement sim.Gpu
}


//___________________________________________________________________________________________________ Time stepping

func(d Gpu) torque(m, h unsafe.Pointer, alpha, dtGilbert float, N int){
  C.gpu_torque((*_C_float)(m), (*_C_float)(h), _C_float(alpha), _C_float(dtGilbert), _C_int(N))
}


func(d Gpu) normalize(m unsafe.Pointer, N int){
  C.gpu_normalize_uniform((*_C_float)(m), _C_int(N))
}

func(d Gpu) normalizeMap(m, normMap unsafe.Pointer, N int){
  C.gpu_normalize_map((*_C_float)(m), (*_C_float)(normMap), _C_int(N))
}


func(d Gpu) eulerStage(m, torque unsafe.Pointer, N int){
  C.gpu_euler_stage((*_C_float)(m), (*_C_float)(torque), _C_int(N))
}

func(d Gpu) semianalStep(m, h unsafe.Pointer, dt, alpha float, N int){
  C.gpu_anal_fw_step_unsafe((*_C_float)(m), (*_C_float)(h), _C_float(dt), _C_float(alpha), _C_int(N))
}

//___________________________________________________________________________________________________ Kernel multiplication


func(d Gpu) kernelMul6(mx, my, mz, kxx, kyy, kzz, kyz, kxz, kxy unsafe.Pointer, nRealNumbers int){
  C.gpu_kernel_mul_complex_inplace_symm(
        (*_C_float)(mx), (*_C_float)(my), (*_C_float)(mz),
        (*_C_float)(kxx), (*_C_float)(kyy), (*_C_float)(kzz),
        (*_C_float)(kyz), (*_C_float)(kxz), (*_C_float)(kxy),
        _C_int(nRealNumbers))
}

//___________________________________________________________________________________________________ Copy-pad


///Copies from a smaller to a larger tensor, not touching the additional space in the destination (typically filled with zero padding)
func(d Gpu) copyPad(source, dest unsafe.Pointer, sourceSize, destSize []int){
  C.gpu_copy_pad_unsafe((*_C_float)(source), (*_C_float)(dest),
                        _C_int(sourceSize[0]), _C_int(sourceSize[1]), _C_int(sourceSize[2]),
                        _C_int(  destSize[0]), _C_int(  destSize[1]), _C_int(  destSize[2]))
}


//Copies from a larger to a smaller tensor, not reading the additional data in the source (typically filled with zero padding or spoiled data)
func(d Gpu) copyUnpad(source, dest unsafe.Pointer, sourceSize, destSize []int){
 C.gpu_copy_unpad_unsafe((*_C_float)(source), (*_C_float)(dest),
                        _C_int(sourceSize[0]), _C_int(sourceSize[1]), _C_int(sourceSize[2]),
                        _C_int(  destSize[0]), _C_int(  destSize[1]), _C_int(  destSize[2]))
}

//___________________________________________________________________________________________________ FFT

/// unsafe creation of C fftPlan
func(d Gpu) newFFTPlan(dataSize, logicSize []int) unsafe.Pointer{
  Csize := (*_C_int)(unsafe.Pointer(&dataSize[0]))
  CpaddedSize := (*_C_int)(unsafe.Pointer(&logicSize[0]))
  return unsafe.Pointer( C.new_gpuFFT3dPlan_padded( Csize, CpaddedSize ) )
}

/// unsafe FFT
func(d Gpu) fftForward(plan unsafe.Pointer, in, out unsafe.Pointer){
  C.gpuFFT3dPlan_forward_unsafe((*_C_gpuFFT3dPlan)(plan), (*_C_float)(in), (*_C_float)(out))
}


/// unsafe FFT
func(d Gpu) fftInverse(plan unsafe.Pointer, in, out unsafe.Pointer){
  C.gpuFFT3dPlan_inverse_unsafe((*_C_gpuFFT3dPlan)(plan), (*_C_float)(in), (*_C_float)(out))
}


// func(d Gpu) (fft *FFT) Normalization() int{
//   return int(C.gpuFFT3dPlan_normalization((*_C_gpuFFT3dPlan)(fft.plan)))
// }


//_______________________________________________________________________________ GPU memory allocation

/**
 * Allocates an array of floats on the GPU.
 * By convention, GPU arrays are represented by an unsafe.Pointer,
 * while host arrays are *float's.
 */
func(d Gpu) newArray(nFloats int) unsafe.Pointer{
  return unsafe.Pointer(C.new_gpu_array(_C_int(nFloats)))
}

/// Copies a number of floats from host to GPU
func(d Gpu) memcpyTo(source *float, dest unsafe.Pointer, nFloats int){
  C.memcpy_to_gpu((*_C_float)(unsafe.Pointer(source)), (*_C_float)(dest), _C_int(nFloats))
}

/// Copies a number of floats from GPU to host
func(d Gpu) memcpyFrom(source unsafe.Pointer, dest *float, nFloats int){
  C.memcpy_from_gpu((*_C_float)(source), (*_C_float)(unsafe.Pointer(dest)), _C_int(nFloats))
}

/// Copies a number of floats from GPU to GPU
func(d Gpu) memcpyOn(source, dest unsafe.Pointer, nFloats int){
  C.memcpy_gpu_to_gpu((*_C_float)(source), (*_C_float)(dest), _C_int(nFloats))
}

/// Gets one float from a GPU array
func(d Gpu) arrayGet(array unsafe.Pointer, index int) float{
  return float(C.gpu_array_get((*_C_float)(array), _C_int(index)))
}

func(d Gpu) arraySet(array unsafe.Pointer, index int, value float){
  C.gpu_array_set((*_C_float)(array), _C_int(index), _C_float(value))
}

func(d Gpu) arrayOffset(array unsafe.Pointer, index int) unsafe.Pointer{
  return unsafe.Pointer(C.gpu_array_offset((*_C_float)(array), _C_int(index)))
} 

//___________________________________________________________________________________________________ GPU Stride

/// The GPU stride in number of floats (!)
func(d Gpu) Stride() int{
  return int(C.gpu_stride_float())
}

/// Takes an array size and returns the smallest multiple of Stride() where the array size fits in
// func(d Gpu) PadToStride(nFloats int) int{
//   return int(C.gpu_pad_to_stride(_C_int(nFloats)));
// }

/// Override the GPU stride, handy for debugging. -1 Means reset to the original GPU stride
func(d Gpu) overrideStride(nFloats int){
  C.gpu_override_stride(_C_int(nFloats))
}

//___________________________________________________________________________________________________ tensor utilities

/// Overwrite n floats with zeros
func(d Gpu) zero(data unsafe.Pointer, nFloats int){
  C.gpu_zero((*_C_float)(data), _C_int(nFloats))
}



/// Print the GPU properties to stdout
func(d Gpu) PrintProperties(){
  C.print_device_properties_stdout()
}

//___________________________________________________________________________________________________ misc

func(d Gpu) String() string{
  return "GPU\n"
}

// TODO does not really belong here but not worth making a new cgo file
func TimerPrintDetail(){
  C.timer_printdetail()
}

//___________________________________________________________________________________________________ go utilities

// func(d Gpu) ToCTensor(t tensor.StoredTensor) *_C_tensor{
//   return C.as_tensorN((*_C_float)(unsafe.Pointer(&(t.List()[0]))), (_C_int)(tensor.Rank(t)), (*_C_int)(unsafe.Pointer(&(t.Size()[0]))) );
// }
// 
// func(d Gpu) ToCGPUTensor(t *Tensor) *_C_tensor{
//   return C.as_tensorN((*_C_float)(t.data), (_C_int)(tensor.Rank(t)), (*_C_int)(unsafe.Pointer(&(t.Size()[0]))) );
// }

// func(d Gpu) assert(b bool){
//   if !b{
//     log.Crash("assertion failed");
//   }
// }
// 
// func(d Gpu) assertEqualSize(sizeA, sizeB []int){
//   assert(len(sizeA) == len(sizeB));
//   for i:=range(sizeA){
//     assert(sizeA[i] == sizeB[i]);
//   }
// }
