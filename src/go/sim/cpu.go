package sim

/*
#include "cpukern.h"

// to allow some (evil but neccesary) pointer arithmetic in go
float* cpu_array_offset(float* array, int index){
    return &array[index];
}
*/
import "C"
import "unsafe"

/**
 * This single file interfaces all the relevant FFTW/cpu functions with go
 * It only wraps the functions, higher level constructs and assetions
 * are in separate files like fft.go, ...
 *
 * @note cgo does not seem to like many cgofiles, so I put everything together here.
 * @author Arne Vansteenkiste
 */

import (
	"fmt"
)

var CPU *Backend = NewBackend(&Cpu{})

type Cpu struct {
	// intentionally empty, but the methods implement sim.Device
}

func (d Cpu) init() {
	C.cpu_init()
}

func (d Cpu) add(a, b unsafe.Pointer, N int) {
	C.cpu_add((*C.float)(a), (*C.float)(b), C.int(N))
}

func (d Cpu) linearCombination(a, b unsafe.Pointer, weightA, weightB float, N int) {
	C.cpu_linear_combination((*C.float)(a), (*C.float)(b), C.float(weightA), C.float(weightB), C.int(N))
}

func (d Cpu)  reduce(operation int, input, output unsafe.Pointer, blocks, threads, N int){
  C.cpu_reduce(C.int(operation), (*C.float)(input), (*C.float)(output), C.int(blocks),  C.int(threads),  C.int(N))
}

func (d Cpu) addConstant(a unsafe.Pointer, cnst float, N int) {
	C.cpu_add_constant((*C.float)(a), C.float(cnst), C.int(N))
}

func (d Cpu) normalize(m unsafe.Pointer, N int) {
	C.cpu_normalize_uniform((*C.float)(m), C.int(N))
}

func (d Cpu) normalizeMap(m, normMap unsafe.Pointer, N int) {
	C.cpu_normalize_map((*C.float)(m), (*C.float)(normMap), C.int(N))
}

func (d Cpu) deltaM(m, h unsafe.Pointer, alpha, dtGilbert float, N int) {
	C.cpu_deltaM((*C.float)(m), (*C.float)(h), C.float(alpha), C.float(dtGilbert), C.int(N))
}

func (d Cpu) semianalStep(m, h unsafe.Pointer, dt, alpha float, order, N int) {
	switch order {
	default:
		panic(fmt.Sprintf("Unknown semianal order:", order))
	case 0:
		C.cpu_anal_fw_step_unsafe((*C.float)(m), (*C.float)(h), C.float(dt), C.float(alpha), C.int(N))
	}
}

//___________________________________________________________________________________________________ Kernel multiplication


func (d Cpu) extractReal(complex, real unsafe.Pointer, NReal int) {
	C.cpu_extract_real((*C.float)(complex), (*C.float)(real), C.int(NReal))
}

func (d Cpu) kernelMul(mx, my, mz, kxx, kyy, kzz, kyz, kxz, kxy unsafe.Pointer, kerneltype, nRealNumbers int) {
	switch kerneltype {
	default:
		panic(fmt.Sprintf("Unknown kernel type:", kerneltype))
	case 6:
		C.cpu_kernelmul6(
			(*C.float)(mx), (*C.float)(my), (*C.float)(mz),
			(*C.float)(kxx), (*C.float)(kyy), (*C.float)(kzz),
			(*C.float)(kyz), (*C.float)(kxz), (*C.float)(kxy),
			C.int(nRealNumbers))
	}

}

//___________________________________________________________________________________________________ Copy-pad


func (d Cpu) copyPadded(source, dest unsafe.Pointer, sourceSize, destSize []int, direction int) {
	switch direction {
	default:
		panic(fmt.Sprintf("Unknown padding direction:", direction))
	case CPY_PAD:
		C.cpu_copy_pad((*C.float)(source), (*C.float)(dest),
			C.int(sourceSize[0]), C.int(sourceSize[1]), C.int(sourceSize[2]),
			C.int(destSize[0]), C.int(destSize[1]), C.int(destSize[2]))
	case CPY_UNPAD:
		C.cpu_copy_unpad((*C.float)(source), (*C.float)(dest),
			C.int(sourceSize[0]), C.int(sourceSize[1]), C.int(sourceSize[2]),
			C.int(destSize[0]), C.int(destSize[1]), C.int(destSize[2]))
	}
}

//___________________________________________________________________________________________________ FFT


// unsafe creation of C fftPlan INPLACE
// TODO outplace, check placeness
func (d Cpu) newFFTPlan(dataSize, logicSize []int) unsafe.Pointer {
	Csize := (*C.int)(unsafe.Pointer(&dataSize[0]))
	CpaddedSize := (*C.int)(unsafe.Pointer(&logicSize[0]))
	return unsafe.Pointer(C.new_cpuFFT3dPlan_inplace(Csize, CpaddedSize))
}

func (d Cpu) fft(plan unsafe.Pointer, in, out unsafe.Pointer, direction int) {
	switch direction {
	default:
		panic(fmt.Sprintf("Unknown FFT direction:", direction))
	case FFT_FORWARD:
		C.cpuFFT3dPlan_forward((*C.cpuFFT3dPlan)(plan), (*C.float)(in), (*C.float)(out))
	case FFT_INVERSE:
		C.cpuFFT3dPlan_inverse((*C.cpuFFT3dPlan)(plan), (*C.float)(in), (*C.float)(out))
	}
}

/// unsafe FFT
// func (d Cpu) fftForward(plan unsafe.Pointer, in, out unsafe.Pointer) {
// 	C.cpuFFT3dPlan_forward((*C.cpuFFT3dPlan)(plan), (*C.float)(in), (*C.float)(out))
// }
//
//
// /// unsafe FFT
// func (d Cpu) fftInverse(plan unsafe.Pointer, in, out unsafe.Pointer) {
// 	C.cpuFFT3dPlan_inverse((*C.cpuFFT3dPlan)(plan), (*C.float)(in), (*C.float)(out))
// }


// func(d Cpu) (fft *FFT) Normalization() int{
//   return int(C.gpuFFT3dPlan_normalization((*C.gpuFFT3dPlan)(fft.plan)))
// }


//_______________________________________________________________________________ GPU memory allocation

// Allocates an array of floats on the CPU.
// By convention, GPU arrays are represented by an unsafe.Pointer,
// while host arrays are *float's.
func (d Cpu) newArray(nFloats int) unsafe.Pointer {
	return unsafe.Pointer(C.new_cpu_array(C.int(nFloats)))
}

func (d Cpu) memcpy(source, dest unsafe.Pointer, nFloats, direction int) {
	C.cpu_memcpy((*C.float)(source), (*C.float)(dest), C.int(nFloats)) //direction is ignored, it's always "CPY_ON" because there is no separate device
}

// ///
// func (d Cpu) memcpyTo(source *float, dest unsafe.Pointer, nFloats int) {
// 	C.cpu_memcpy((*C.float)(unsafe.Pointer(source)), (*C.float)(dest), C.int(nFloats))
// }
//
// ///
// func (d Cpu) memcpyFrom(source unsafe.Pointer, dest *float, nFloats int) {
// 	C.cpu_memcpy((*C.float)(source), (*C.float)(unsafe.Pointer(dest)), C.int(nFloats))
// }
//
// ///
// func (d Cpu) memcpyOn(source, dest unsafe.Pointer, nFloats int) {
// 	C.cpu_memcpy((*C.float)(source), (*C.float)(dest), C.int(nFloats))
//}

/// Gets one float from a GPU array
// func (d Cpu) arrayGet(array unsafe.Pointer, index int) float {
// 	return float(C.cpu_array_get((*C.float)(array), C.int(index)))
// }
//
// func (d Cpu) arraySet(array unsafe.Pointer, index int, value float) {
// 	C.cpu_array_set((*C.float)(array), C.int(index), C.float(value))
// }

func (d Cpu) arrayOffset(array unsafe.Pointer, index int) unsafe.Pointer {
	return unsafe.Pointer(C.cpu_array_offset((*C.float)(array), C.int(index)))
}

//___________________________________________________________________________________________________ GPU Stride

// The GPU stride in number of floats (!)
func (d Cpu) Stride() int {
	return int(C.cpu_stride_float())
}

// Takes an array size and returns the smallest multiple of Stride() where the array size fits in
// func(d Cpu) PadToStride(nFloats int) int{
//   return int(C.cpu_pad_to_stride(C.int(nFloats)));
// }

// Override the GPU stride, handy for debugging. -1 Means reset to the original GPU stride
func (d Cpu) overrideStride(nFloats int) {
	C.cpu_override_stride(C.int(nFloats))
}

//___________________________________________________________________________________________________ tensor utilities

/// Overwrite n floats with zeros
func (d Cpu) zero(data unsafe.Pointer, nFloats int) {
	C.cpu_zero((*C.float)(data), C.int(nFloats))
}

func (d Cpu) UsedMem() uint64 {
	return 0 // meh
}

// Print the GPU properties to stdout
func (d Cpu) PrintProperties() {
	C.cpu_print_properties_stdout()
}

// //___________________________________________________________________________________________________ misc

func (d Cpu) String() string {
	return "CPU"
}

func (d Cpu) TimerPrintDetail() {
	//C.timer_printdetail()
	fmt.Println("meh...")
}
