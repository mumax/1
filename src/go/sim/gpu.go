package sim

/*
#include "gpukern.h"
#include "timer.h"

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

import (
	"fmt"
)

var GPU *Backend = NewBackend(&Gpu{})

type Gpu struct {
	// intentionally empty, but the methods implement sim.Device
}

func (d Gpu) init() {
	C.gpu_init()
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

func (d Gpu) semianalStep(m, h unsafe.Pointer, dt, alpha float, order, N int) {
	switch order {
	default:
		panic(fmt.Sprintf("Unknown semianal order:", order))
	case 0:
		C.gpu_anal_fw_step_unsafe((*C.float)(m), (*C.float)(h), C.float(dt), C.float(alpha), C.int(N))
	}
}


func (d Gpu) kernelMul(mx, my, mz, kxx, kyy, kzz, kyz, kxz, kxy unsafe.Pointer, kerneltype, nRealNumbers int) {
	switch kerneltype {
	default:
		panic(fmt.Sprintf("Unknown kernel type:", kerneltype))
	case 6:
		C.gpu_kernelmul6(
			(*C.float)(mx), (*C.float)(my), (*C.float)(mz),
			(*C.float)(kxx), (*C.float)(kyy), (*C.float)(kzz),
			(*C.float)(kyz), (*C.float)(kxz), (*C.float)(kxy),
			C.int(nRealNumbers))
	}
}


func (d Gpu) copyPadded(source, dest unsafe.Pointer, sourceSize, destSize []int, direction int) {
	switch direction {
	default:
		panic(fmt.Sprintf("Unknown padding direction:", direction))
	case CPY_PAD:
		C.gpu_copy_pad((*C.float)(source), (*C.float)(dest),
			C.int(sourceSize[0]), C.int(sourceSize[1]), C.int(sourceSize[2]),
			C.int(destSize[0]), C.int(destSize[1]), C.int(destSize[2]))
	case CPY_UNPAD:
		C.gpu_copy_unpad((*C.float)(source), (*C.float)(dest),
			C.int(sourceSize[0]), C.int(sourceSize[1]), C.int(sourceSize[2]),
			C.int(destSize[0]), C.int(destSize[1]), C.int(destSize[2]))
	}
}


func (d Gpu) newFFTPlan(dataSize, logicSize []int) unsafe.Pointer {
	Csize := (*C.int)(unsafe.Pointer(&dataSize[0]))
	CpaddedSize := (*C.int)(unsafe.Pointer(&logicSize[0]))
	return unsafe.Pointer(C.new_gpuFFT3dPlan_padded(Csize, CpaddedSize))
}


func (d Gpu) fft(plan unsafe.Pointer, in, out unsafe.Pointer, direction int) {
	switch direction {
	default:
		panic(fmt.Sprintf("Unknown FFT direction:", direction))
	case FFT_FORWARD:
		C.gpuFFT3dPlan_forward((*C.gpuFFT3dPlan)(plan), (*C.float)(in), (*C.float)(out))
	case FFT_INVERSE:
		C.gpuFFT3dPlan_inverse((*C.gpuFFT3dPlan)(plan), (*C.float)(in), (*C.float)(out))
	}
}


func (d Gpu) newArray(nFloats int) unsafe.Pointer {
	return unsafe.Pointer(C.new_gpu_array(C.int(nFloats)))
}


func (d Gpu) memcpy(source, dest unsafe.Pointer, nFloats, direction int) {
	C.memcpy_gpu_dir((*C.float)(unsafe.Pointer(source)), (*C.float)(dest), C.int(nFloats), C.int(direction))
}


func (d Gpu) arrayOffset(array unsafe.Pointer, index int) unsafe.Pointer {
	return unsafe.Pointer(C.gpu_array_offset((*C.float)(array), C.int(index)))
}

func (d Gpu) Stride() int {
	return int(C.gpu_stride_float())
}

func (d Gpu) overrideStride(nFloats int) {
	C.gpu_override_stride(C.int(nFloats))
}

func (d Gpu) zero(data unsafe.Pointer, nFloats int) {
	C.gpu_zero((*C.float)(data), C.int(nFloats))
}

func (d Gpu) UsedMem() uint64 {
	return uint64(C.gpu_usedmem())
}

func (d Gpu) PrintProperties() {
	C.gpu_print_properties_stdout()
}


func (d Gpu) String() string {
	return "GPU"
}

func (d Gpu) TimerPrintDetail() {
	C.timer_printdetail()
}
