package sim

/*
#include "gpukern.h"

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

import ()

var GPU Backend = Backend{Gpu{}}

type Gpu struct {
	// intentionally empty, but the methods implement sim.Gpu
}


func (d Gpu) add(a, b unsafe.Pointer, N int) {
	C.gpu_add((*C.float)(a), (*C.float)(b), C.int(N))
}

func (d Gpu) linearCombination(a, b unsafe.Pointer, weightA, weightB float, N int) {
	C.gpu_linear_combination((*C.float)(a), (*C.float)(b), C.float(weightA), C.float(weightB), C.int(N))
}

func (d Gpu) addConstant(a unsafe.Pointer, cnst float, N int) {
	C.gpu_add_constant((*C.float)(a), C.float(cnst), C.int(N))
}

func (d Gpu) normalize(m unsafe.Pointer, N int) {
	C.gpu_normalize_uniform((*C.float)(m), C.int(N))
}

func (d Gpu) normalizeMap(m, normMap unsafe.Pointer, N int) {
	C.gpu_normalize_map((*C.float)(m), (*C.float)(normMap), C.int(N))
}

func (d Gpu) deltaM(m, h unsafe.Pointer, alpha, dtGilbert float, N int) {
	C.gpu_deltaM((*C.float)(m), (*C.float)(h), C.float(alpha), C.float(dtGilbert), C.int(N))
}

func (d Gpu) semianalStep(m, h unsafe.Pointer, dt, alpha float, N int) {
	C.gpu_anal_fw_step_unsafe((*C.float)(m), (*C.float)(h), C.float(dt), C.float(alpha), C.int(N))
}

//___________________________________________________________________________________________________ Kernel multiplication


func (d Gpu) kernelMul6(mx, my, mz, kxx, kyy, kzz, kyz, kxz, kxy unsafe.Pointer, nRealNumbers int) {
	C.gpu_kernelmul6(
		(*C.float)(mx), (*C.float)(my), (*C.float)(mz),
		(*C.float)(kxx), (*C.float)(kyy), (*C.float)(kzz),
		(*C.float)(kyz), (*C.float)(kxz), (*C.float)(kxy),
		C.int(nRealNumbers))
}

//___________________________________________________________________________________________________ Copy-pad


///Copies from a smaller to a larger tensor, not touching the additional space in the destination (typically filled with zero padding)
func (d Gpu) copyPad(source, dest unsafe.Pointer, sourceSize, destSize []int) {
	C.gpu_copy_pad((*C.float)(source), (*C.float)(dest),
		C.int(sourceSize[0]), C.int(sourceSize[1]), C.int(sourceSize[2]),
		C.int(destSize[0]), C.int(destSize[1]), C.int(destSize[2]))
}


//Copies from a larger to a smaller tensor, not reading the additional data in the source (typically filled with zero padding or spoiled data)
func (d Gpu) copyUnpad(source, dest unsafe.Pointer, sourceSize, destSize []int) {
	C.gpu_copy_unpad((*C.float)(source), (*C.float)(dest),
		C.int(sourceSize[0]), C.int(sourceSize[1]), C.int(sourceSize[2]),
		C.int(destSize[0]), C.int(destSize[1]), C.int(destSize[2]))
}

//___________________________________________________________________________________________________ FFT

/// unsafe creation of C fftPlan
func (d Gpu) newFFTPlan(dataSize, logicSize []int) unsafe.Pointer {
	Csize := (*C.int)(unsafe.Pointer(&dataSize[0]))
	CpaddedSize := (*C.int)(unsafe.Pointer(&logicSize[0]))
	return unsafe.Pointer(C.new_gpuFFT3dPlan_padded(Csize, CpaddedSize))
}

/// unsafe FFT
func (d Gpu) fftForward(plan unsafe.Pointer, in, out unsafe.Pointer) {
	C.gpuFFT3dPlan_forward((*C.gpuFFT3dPlan)(plan), (*C.float)(in), (*C.float)(out))
}


/// unsafe FFT
func (d Gpu) fftInverse(plan unsafe.Pointer, in, out unsafe.Pointer) {
	C.gpuFFT3dPlan_inverse((*C.gpuFFT3dPlan)(plan), (*C.float)(in), (*C.float)(out))
}


// func(d Gpu) (fft *FFT) Normalization() int{
//   return int(C.gpuFFT3dPlan_normalization((*C.gpuFFT3dPlan)(fft.plan)))
// }


//_______________________________________________________________________________ GPU memory allocation

/**
 * Allocates an array of floats on the GPU.
 * By convention, GPU arrays are represented by an unsafe.Pointer,
 * while host arrays are *float's.
 */
func (d Gpu) newArray(nFloats int) unsafe.Pointer {
	return unsafe.Pointer(C.new_gpu_array(C.int(nFloats)))
}

/// Copies a number of floats from host to GPU
func (d Gpu) memcpyTo(source *float, dest unsafe.Pointer, nFloats int) {
	C.memcpy_to_gpu((*C.float)(unsafe.Pointer(source)), (*C.float)(dest), C.int(nFloats))
}

/// Copies a number of floats from GPU to host
func (d Gpu) memcpyFrom(source unsafe.Pointer, dest *float, nFloats int) {
	C.memcpy_from_gpu((*C.float)(source), (*C.float)(unsafe.Pointer(dest)), C.int(nFloats))
}

/// Copies a number of floats from GPU to GPU
func (d Gpu) memcpyOn(source, dest unsafe.Pointer, nFloats int) {
	C.memcpy_gpu_to_gpu((*C.float)(source), (*C.float)(dest), C.int(nFloats))
}

/// Gets one float from a GPU array
func (d Gpu) arrayGet(array unsafe.Pointer, index int) float {
	return float(C.gpu_array_get((*C.float)(array), C.int(index)))
}

func (d Gpu) arraySet(array unsafe.Pointer, index int, value float) {
	C.gpu_array_set((*C.float)(array), C.int(index), C.float(value))
}

func (d Gpu) arrayOffset(array unsafe.Pointer, index int) unsafe.Pointer {
	return unsafe.Pointer(C.gpu_array_offset((*C.float)(array), C.int(index)))
}

//___________________________________________________________________________________________________ GPU Stride

/// The GPU stride in number of floats (!)
func (d Gpu) Stride() int {
	return int(C.gpu_stride_float())
}

/// Takes an array size and returns the smallest multiple of Stride() where the array size fits in
// func(d Gpu) PadToStride(nFloats int) int{
//   return int(C.gpu_pad_to_stride(C.int(nFloats)));
// }

/// Override the GPU stride, handy for debugging. -1 Means reset to the original GPU stride
func (d Gpu) overrideStride(nFloats int) {
	C.gpu_override_stride(C.int(nFloats))
}

//___________________________________________________________________________________________________ tensor utilities

/// Overwrite n floats with zeros
func (d Gpu) zero(data unsafe.Pointer, nFloats int) {
	C.gpu_zero((*C.float)(data), C.int(nFloats))
}


/// Print the GPU properties to stdout
func (d Gpu) PrintProperties() {
	C.gpu_print_properties_stdout()
}

//___________________________________________________________________________________________________ misc

func (d Gpu) String() string {
	return "GPU"
}

// func TimerPrintDetail(){
//   C.timer_printdetail()
// }

//___________________________________________________________________________________________________ go utilities

// func(d Gpu) ToCTensor(t tensor.StoredTensor) *_C_tensor{
//   return C.as_tensorN((*C.float)(unsafe.Pointer(&(t.List()[0]))), (C.int)(tensor.Rank(t)), (*C.int)(unsafe.Pointer(&(t.Size()[0]))) );
// }
//
// func(d Gpu) ToCGPUTensor(t *Tensor) *_C_tensor{
//   return C.as_tensorN((*C.float)(t.data), (C.int)(tensor.Rank(t)), (*C.int)(unsafe.Pointer(&(t.Size()[0]))) );
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
