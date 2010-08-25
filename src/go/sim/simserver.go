package sim

import (
	"unsafe"
	"rpc"
	"net"
	"http"
	"os"
	"fmt"
)

type DeviceServer struct {
	Device
	port string
}

// TODO: to be put in simulation main
func SimServerMain() {
	Verbosity = 3

	server := &DeviceServer{CPU.Device, ":2527"}
	rpc.Register(server)
	rpc.HandleHTTP()
	listener, err := net.Listen("tcp", server.port)
	if err != nil {
		panic(err)
	}
	fmt.Println("Listening on port", server.port)
	http.Serve(listener, nil)
}


func (s *DeviceServer) Add(in *AddArgs, out *Void) os.Error {
	s.add(unsafe.Pointer(in.A), unsafe.Pointer(in.B), in.N)
	return nil
}

func (s *DeviceServer) LinearCombination(in *LinearCombinationArgs, out *Void) os.Error {
	s.linearCombination(unsafe.Pointer(in.A), unsafe.Pointer(in.B), in.WeightA, in.WeightB, in.N)
	return nil
}

func (s *DeviceServer) AddConstant(in *AddConstantArgs, out *Void) os.Error {
	s.addConstant(unsafe.Pointer(in.A), in.Cnst, in.N)
	return nil
}

func (s *DeviceServer) Normalize(in *NormalizeArgs, out *Void) os.Error {
	s.normalize(unsafe.Pointer(in.M), in.N)
	return nil
}

func (s *DeviceServer) NormalizeMap(in *NormalizeMapArgs, out *Void) os.Error {
	s.normalizeMap(unsafe.Pointer(in.M), unsafe.Pointer(in.NormMap), in.N)
	return nil
}

func (s *DeviceServer) DeltaM(in *DeltaMArgs, out *Void) os.Error {
	s.deltaM(unsafe.Pointer(in.M), unsafe.Pointer(in.H), in.Alpha, in.DtGilbert, in.N)
	return nil
}

/*/*
func (s *DeviceServer) semianalStep(in * , out *Void) {
  C.gpu_anal_fw_step_unsafe((*C.float)(m), (*C.float)(h), C.float(dt), C.float(alpha), C.int(N))
}

//___________________________________________________________________________________________________ Kernel multiplication


func (s *DeviceServer) extractReal(in * , out *Void) {
  C.gpu_extract_real((*C.float)(complex), (*C.float)(real), C.int(NReal))
}

func (s *DeviceServer) kernelMul6(in * , out *Void) {
  C.gpu_kernelmul6(
    (*C.float)(mx), (*C.float)(my), (*C.float)(mz),
    (*C.float)(kxx), (*C.float)(kyy), (*C.float)(kzz),
    (*C.float)(kyz), (*C.float)(kxz), (*C.float)(kxy),
    C.int(nRealNumbers))
}

//___________________________________________________________________________________________________ Copy-pad


///Copies from a smaller to a larger tensor, not touching the additional space in the destination (typically filled with zero padding)
func (s *DeviceServer) copyPad(in * , out *Void) {
  C.gpu_copy_pad((*C.float)(source), (*C.float)(dest),
    C.int(sourceSize[0]), C.int(sourceSize[1]), C.int(sourceSize[2]),
    C.int(destSize[0]), C.int(destSize[1]), C.int(destSize[2]))
}


//Copies from a larger to a smaller tensor, not reading the additional data in the source (typically filled with zero padding or spoiled data)
func (s *DeviceServer) copyUnpad(in * , out *Void) {
  C.gpu_copy_unpad((*C.float)(source), (*C.float)(dest),
    C.int(sourceSize[0]), C.int(sourceSize[1]), C.int(sourceSize[2]),
    C.int(destSize[0]), C.int(destSize[1]), C.int(destSize[2]))
}

//___________________________________________________________________________________________________ FFT

/// unsafe creation of C fftPlan
func (s *DeviceServer) newFFTPlan(in * , out *Void) unsafe.Pointer {
  Csize := (*C.int)(unsafe.Pointer(&dataSize[0]))
  CpaddedSize := (*C.int)(unsafe.Pointer(&logicSize[0]))
  return unsafe.Pointer(C.new_gpuFFT3dPlan_padded(Csize, CpaddedSize))
}

/// unsafe FFT
func (s *DeviceServer) fftForward(in * , out *Void) {
  C.gpuFFT3dPlan_forward((*C.gpuFFT3dPlan)(plan), (*C.float)(in), (*C.float)(out))
}


/// unsafe FFT
func (s *DeviceServer) fftInverse(in * , out *Void) {
  C.gpuFFT3dPlan_inverse((*C.gpuFFT3dPlan)(plan), (*C.float)(in), (*C.float)(out))
}


//_______________________________________________________________________________ GPU memory allocation


// Allocates an array of floats on the GPU.
// By convention, GPU arrays are represented by an unsafe.Pointer,
// while host arrays are *float's.
func (s *DeviceServer) NewArray(in * , out *Void) unsafe.Pointer {
//   return unsafe.Pointer(C.new_gpu_array(C.int(nFloats)))
  return nil
}

/// Copies a number of floats from host to the device
func (s *DeviceServer) MemcpyTo(in * , out *Void) {
//   C.memcpy_to_gpu((*C.float)(unsafe.Pointer(source)), (*C.float)(dest), C.int(nFloats))
return nil
}

/// Copies a number of floats from the device to host
func (s *DeviceServer) MemcpyFrom(in * , out *Void) {
//   C.memcpy_from_gpu((*C.float)(source), (*C.float)(unsafe.Pointer(dest)), C.int(nFloats))
return nil
}

/// Copies a number of floats on the device
func (s *DeviceServer) MemcpyOn(in * , out *Void) {
//   C.memcpy_gpu_to_gpu((*C.float)(source), (*C.float)(dest), C.int(nFloats))
return nil
}

/*
/// Gets one float from a GPU array
func (s *DeviceServer) arrayGet(in * , out *Void) float {
  return float(C.gpu_array_get((*C.float)(array), C.int(index)))
}

func (s *DeviceServer) arraySet(in * , out *Void) {
  C.gpu_array_set((*C.float)(array), C.int(index), C.float(value))
}

func (s *DeviceServer) arrayOffset(in * , out *Void) unsafe.Pointer {
  return unsafe.Pointer(C.gpu_array_offset((*C.float)(array), C.int(index)))
}

//___________________________________________________________________________________________________ GPU Stride

/// The GPU stride in number of floats (!)
func (s *DeviceServer) Stride(in * , out *Void) int {
  return int(C.gpu_stride_float())
}

/// Takes an array size and returns the smallest multiple of Stride() where the array size fits in
// func(s *DeviceServer) PadToStride(nFloats int) int{
//   return int(C.gpu_pad_to_stride(C.int(nFloats)));
// }

/// Override the GPU stride, handy for debugging. -1 Means reset to the original GPU stride
func (s *DeviceServer) overrideStride(in * , out *Void) {
  C.gpu_override_stride(C.int(nFloats))
}

//___________________________________________________________________________________________________ tensor utilities

/// Overwrite n floats with zeros
func (s *DeviceServer) zero(in * , out *Void) {
  C.gpu_zero((*C.float)(data), C.int(nFloats))
}


/// Print the GPU properties to stdout
func (s *DeviceServer) PrintProperties() {
  C.gpu_print_properties_stdout()
}

//___________________________________________________________________________________________________ misc*/
func (s *DeviceServer) String() string {
	return "Remote"
}
