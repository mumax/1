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

func (d Gpu) add(a, b uintptr, N int) {
	C.gpu_add((*C.float)(unsafe.Pointer(a)), (*C.float)(unsafe.Pointer(b)), C.int(N))
}

func (d Gpu) madd(a uintptr, cnst float, b uintptr, N int) {
	C.gpu_madd((*C.float)(unsafe.Pointer(a)), C.float(cnst), (*C.float)(unsafe.Pointer(b)), C.int(N))
}

func (d Gpu) linearCombination(a, b uintptr, weightA, weightB float, N int) {
	C.gpu_linear_combination((*C.float)(unsafe.Pointer(a)), (*C.float)(unsafe.Pointer(b)), C.float(weightA), C.float(weightB), C.int(N))
}

func (d Gpu) addConstant(a uintptr, cnst float, N int) {
	C.gpu_add_constant((*C.float)(unsafe.Pointer(a)), C.float(cnst), C.int(N))
}

func (d Gpu) reduce(operation int, input, output uintptr, buffer *float, blocks, threads, N int) float {
	return float(C.gpu_reduce(C.int(operation), (*C.float)(unsafe.Pointer(input)), (*C.float)(unsafe.Pointer(output)), (*C.float)(unsafe.Pointer(buffer)), C.int(blocks), C.int(threads), C.int(N)))
}

func (d Gpu) normalize(m uintptr, N int) {
	C.gpu_normalize_uniform((*C.float)(unsafe.Pointer(m)), C.int(N))
}

func (d Gpu) normalizeMap(m, normMap uintptr, N int) {
	C.gpu_normalize_map((*C.float)(unsafe.Pointer(m)), (*C.float)(unsafe.Pointer(normMap)), C.int(N))
}

func (d Gpu) deltaM(m, h uintptr, alpha, dtGilbert float, N int) {
	C.gpu_deltaM((*C.float)(unsafe.Pointer(m)), (*C.float)(unsafe.Pointer(h)), C.float(alpha), C.float(dtGilbert), C.int(N))
}

func (d Gpu) semianalStep(m, h uintptr, dt, alpha float, order, N int) {
	switch order {
	default:
		panic(fmt.Sprintf("Unknown semianal order:", order))
	case 0:
		C.gpu_anal_fw_step_unsafe((*C.float)(unsafe.Pointer(m)), (*C.float)(unsafe.Pointer(h)), C.float(dt), C.float(alpha), C.int(N))
	}
}


func (d Gpu) kernelMul(mx, my, mz, kxx, kyy, kzz, kyz, kxz, kxy uintptr, kerneltype, nRealNumbers int) {
	switch kerneltype {
	default:
		panic(fmt.Sprintf("Unknown kernel type:", kerneltype))
	case 6:
		C.gpu_kernelmul6(
			(*C.float)(unsafe.Pointer(mx)), (*C.float)(unsafe.Pointer(my)), (*C.float)(unsafe.Pointer(mz)),
			(*C.float)(unsafe.Pointer(kxx)), (*C.float)(unsafe.Pointer(kyy)), (*C.float)(unsafe.Pointer(kzz)),
			(*C.float)(unsafe.Pointer(kyz)), (*C.float)(unsafe.Pointer(kxz)), (*C.float)(unsafe.Pointer(kxy)),
			C.int(nRealNumbers))
	}
}


func (d Gpu) copyPadded(source, dest uintptr, sourceSize, destSize []int, direction int) {
	switch direction {
	default:
		panic(fmt.Sprintf("Unknown padding direction:", direction))
	case CPY_PAD:
		C.gpu_copy_pad((*C.float)(unsafe.Pointer(source)), (*C.float)(unsafe.Pointer(dest)),
			C.int(sourceSize[0]), C.int(sourceSize[1]), C.int(sourceSize[2]),
			C.int(destSize[0]), C.int(destSize[1]), C.int(destSize[2]))
	case CPY_UNPAD:
		C.gpu_copy_unpad((*C.float)(unsafe.Pointer(source)), (*C.float)(unsafe.Pointer(dest)),
			C.int(sourceSize[0]), C.int(sourceSize[1]), C.int(sourceSize[2]),
			C.int(destSize[0]), C.int(destSize[1]), C.int(destSize[2]))
	}
}


func (d Gpu) newFFTPlan(dataSize, logicSize []int) uintptr {
	Csize := (*C.int)(unsafe.Pointer(&dataSize[0]))
	CpaddedSize := (*C.int)(unsafe.Pointer(&logicSize[0]))
	return uintptr(unsafe.Pointer(C.new_gpuFFT3dPlanArne_padded(Csize, CpaddedSize)))
}


func (d Gpu) fft(plan uintptr, in, out uintptr, direction int) {
	switch direction {
	default:
		panic(fmt.Sprintf("Unknown FFT direction:", direction))
	case FFT_FORWARD:
		C.gpuFFT3dPlanArne_forward((*C.gpuFFT3dPlanArne)(unsafe.Pointer(plan)), (*C.float)(unsafe.Pointer(in)), (*C.float)(unsafe.Pointer(out)))
	case FFT_INVERSE:
		C.gpuFFT3dPlanArne_inverse((*C.gpuFFT3dPlanArne)(unsafe.Pointer(plan)), (*C.float)(unsafe.Pointer(in)), (*C.float)(unsafe.Pointer(out)))
	}
}


func (d Gpu) newArray(nFloats int) uintptr {
	return uintptr(unsafe.Pointer(C.new_gpu_array(C.int(nFloats))))
}


func (d Gpu) memcpy(source, dest uintptr, nFloats, direction int) {
	C.memcpy_gpu_dir((*C.float)(unsafe.Pointer(source)), (*C.float)(unsafe.Pointer(dest)), C.int(nFloats), C.int(direction))
}


func (d Gpu) arrayOffset(array uintptr, index int) uintptr {
	return uintptr(unsafe.Pointer(C.gpu_array_offset((*C.float)(unsafe.Pointer(array)), C.int(index))))
}

func (d Gpu) Stride() int {
	return int(C.gpu_stride_float())
}

func (d Gpu) overrideStride(nFloats int) {
	C.gpu_override_stride(C.int(nFloats))
}

func (d Gpu) zero(data uintptr, nFloats int) {
	C.gpu_zero((*C.float)(unsafe.Pointer(data)), C.int(nFloats))
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
