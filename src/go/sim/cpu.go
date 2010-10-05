//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

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
	"os"
)

var CPU *Backend = NewBackend(&Cpu{})

type Cpu struct {
	// intentionally empty, but the methods implement sim.Device
}

func (d Cpu) init() {
	C.cpu_init()
}

func (d Cpu) setDevice(devid int) {
	fmt.Fprintln(os.Stderr, "setDevice(", devid, ") has no effect on CPU")
}

func (d Cpu) add(a, b uintptr, N int) {
	C.cpu_add((*C.float)(unsafe.Pointer(a)), (*C.float)(unsafe.Pointer(b)), C.int(N))
}

func (d Cpu) madd(a uintptr, cnst float, b uintptr, N int) {
	C.cpu_madd((*C.float)(unsafe.Pointer(a)), C.float(cnst), (*C.float)(unsafe.Pointer(b)), C.int(N))
}

func (d Cpu) linearCombination(a, b uintptr, weightA, weightB float, N int) {
	C.cpu_linear_combination((*C.float)(unsafe.Pointer(a)), (*C.float)(unsafe.Pointer(b)), C.float(weightA), C.float(weightB), C.int(N))
}

func (d Cpu) reduce(operation int, input, output uintptr, buffer *float, blocks, threads, N int) float {
	//C.cpu_reduce(C.int(operation), (*C.float)(input), (*C.float)(output), C.int(blocks), C.int(threads), C.int(N))
	panic("unimplemented")
}

func (d Cpu) addConstant(a uintptr, cnst float, N int) {
	C.cpu_add_constant((*C.float)(unsafe.Pointer(a)), C.float(cnst), C.int(N))
}

func (d Cpu) normalize(m uintptr, N int) {
	C.cpu_normalize_uniform((*C.float)(unsafe.Pointer(m)), C.int(N))
}

func (d Cpu) normalizeMap(m, normMap uintptr, N int) {
	C.cpu_normalize_map((*C.float)(unsafe.Pointer(m)), (*C.float)(unsafe.Pointer(normMap)), C.int(N))
}

func (d Cpu) deltaM(m, h uintptr, alpha, dtGilbert float, N int) {
	C.cpu_deltaM((*C.float)(unsafe.Pointer(m)), (*C.float)(unsafe.Pointer(h)), C.float(alpha), C.float(dtGilbert), C.int(N))
}

func (d Cpu) semianalStep(m, h uintptr, dt, alpha float, order, N int) {
	switch order {
	default:
		panic(fmt.Sprintf("Unknown semianal order:", order))
	case 0:
		C.cpu_anal_fw_step_unsafe((*C.float)(unsafe.Pointer(m)), (*C.float)(unsafe.Pointer(h)), C.float(dt), C.float(alpha), C.int(N))
	}
}

//___________________________________________________________________________________________________ Kernel multiplication


func (d Cpu) extractReal(complex, real uintptr, NReal int) {
	C.cpu_extract_real((*C.float)(unsafe.Pointer(complex)), (*C.float)(unsafe.Pointer(real)), C.int(NReal))
}

func (d Cpu) kernelMul(mx, my, mz, kxx, kyy, kzz, kyz, kxz, kxy uintptr, kerneltype, nRealNumbers int) {
	switch kerneltype {
	default:
		panic(fmt.Sprintf("Unknown kernel type:", kerneltype))
	case 6:
		C.cpu_kernelmul6(
			(*C.float)(unsafe.Pointer(mx)), (*C.float)(unsafe.Pointer(my)), (*C.float)(unsafe.Pointer(mz)),
			(*C.float)(unsafe.Pointer(kxx)), (*C.float)(unsafe.Pointer(kyy)), (*C.float)(unsafe.Pointer(kzz)),
			(*C.float)(unsafe.Pointer(kyz)), (*C.float)(unsafe.Pointer(kxz)), (*C.float)(unsafe.Pointer(kxy)),
			C.int(nRealNumbers))
	}

}

//___________________________________________________________________________________________________ Copy-pad


func (d Cpu) copyPadded(source, dest uintptr, sourceSize, destSize []int, direction int) {
	switch direction {
	default:
		panic(fmt.Sprintf("Unknown padding direction:", direction))
	case CPY_PAD:
		C.cpu_copy_pad((*C.float)(unsafe.Pointer(source)), (*C.float)(unsafe.Pointer(dest)),
			C.int(sourceSize[0]), C.int(sourceSize[1]), C.int(sourceSize[2]),
			C.int(destSize[0]), C.int(destSize[1]), C.int(destSize[2]))
	case CPY_UNPAD:
		C.cpu_copy_unpad((*C.float)(unsafe.Pointer(source)), (*C.float)(unsafe.Pointer(dest)),
			C.int(sourceSize[0]), C.int(sourceSize[1]), C.int(sourceSize[2]),
			C.int(destSize[0]), C.int(destSize[1]), C.int(destSize[2]))
	}
}

//___________________________________________________________________________________________________ FFT


// unsafe creation of C fftPlan INPLACE
// TODO outplace, check placeness
func (d Cpu) newFFTPlan(dataSize, logicSize []int) uintptr {
	Csize := (*C.int)(unsafe.Pointer(&dataSize[0]))
	CpaddedSize := (*C.int)(unsafe.Pointer(&logicSize[0]))
	return uintptr(unsafe.Pointer(C.new_cpuFFT3dPlan_inplace(Csize, CpaddedSize)))
}

func (d Cpu) fft(plan uintptr, in, out uintptr, direction int) {
	switch direction {
	default:
		panic(fmt.Sprintf("Unknown FFT direction:", direction))
	case FFT_FORWARD:
		C.cpuFFT3dPlan_forward((*C.cpuFFT3dPlan)(unsafe.Pointer(plan)), (*C.float)(unsafe.Pointer(in)), (*C.float)(unsafe.Pointer(out)))
	case FFT_INVERSE:
		C.cpuFFT3dPlan_inverse((*C.cpuFFT3dPlan)(unsafe.Pointer(plan)), (*C.float)(unsafe.Pointer(in)), (*C.float)(unsafe.Pointer(out)))
	}
}

/// unsafe FFT
// func (d Cpu) fftForward(plan uintptr, in, out uintptr) {
// 	C.cpuFFT3dPlan_forward((*C.cpuFFT3dPlan)(plan), (*C.float)(in), (*C.float)(out))
// }
//
//
// /// unsafe FFT
// func (d Cpu) fftInverse(plan uintptr, in, out uintptr) {
// 	C.cpuFFT3dPlan_inverse((*C.cpuFFT3dPlan)(plan), (*C.float)(in), (*C.float)(out))
// }


// func(d Cpu) (fft *FFT) Normalization() int{
//   return int(C.gpuFFT3dPlan_normalization((*C.gpuFFT3dPlan)(fft.plan)))
// }


//_______________________________________________________________________________ GPU memory allocation

// Allocates an array of floats on the CPU.
// By convention, GPU arrays are represented by an uintptr,
// while host arrays are *float's.
func (d Cpu) newArray(nFloats int) uintptr {
	return uintptr(unsafe.Pointer(C.new_cpu_array(C.int(nFloats))))
}

func (d Cpu) freeArray(ptr uintptr) {
	C.free_cpu_array((*C.float)(unsafe.Pointer(ptr)))
}

func (d Cpu) memcpy(source, dest uintptr, nFloats, direction int) {
	C.cpu_memcpy((*C.float)(unsafe.Pointer(source)), (*C.float)(unsafe.Pointer(dest)), C.int(nFloats)) //direction is ignored, it's always "CPY_ON" because there is no separate device
}

// ///
// func (d Cpu) memcpyTo(source *float, dest uintptr, nFloats int) {
// 	C.cpu_memcpy((*C.float)(unsafe.Pointer(source)), (*C.float)(dest), C.int(nFloats))
// }
//
// ///
// func (d Cpu) memcpyFrom(source uintptr, dest *float, nFloats int) {
// 	C.cpu_memcpy((*C.float)(source), (*C.float)(unsafe.Pointer(dest)), C.int(nFloats))
// }
//
// ///
// func (d Cpu) memcpyOn(source, dest uintptr, nFloats int) {
// 	C.cpu_memcpy((*C.float)(source), (*C.float)(dest), C.int(nFloats))
//}

/// Gets one float from a GPU array
// func (d Cpu) arrayGet(array uintptr, index int) float {
// 	return float(C.cpu_array_get((*C.float)(array), C.int(index)))
// }
//
// func (d Cpu) arraySet(array uintptr, index int, value float) {
// 	C.cpu_array_set((*C.float)(array), C.int(index), C.float(value))
// }

func (d Cpu) arrayOffset(array uintptr, index int) uintptr {
	return uintptr(unsafe.Pointer(C.cpu_array_offset((*C.float)(unsafe.Pointer(array)), C.int(index))))
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
func (d Cpu) zero(data uintptr, nFloats int) {
	C.cpu_zero((*C.float)(unsafe.Pointer(data)), C.int(nFloats))
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
