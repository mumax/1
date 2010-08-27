package sim

import (
	"unsafe"
	"rpc"
	"net"
	"http"
	"os"
	"fmt"
)

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

type DeviceServer struct {
	dev  Device // We do not embed to avoid the dev methods to be exported by rpc
	port string
}

func (s *DeviceServer) Init(in, out *Void) {
	s.dev.init()
}

func (s *DeviceServer) Add(in *AddArgs, out *Void) os.Error {
	s.dev.add(unsafe.Pointer(in.A), unsafe.Pointer(in.B), in.N)
	return nil
}

func (s *DeviceServer) LinearCombination(in *LinearCombinationArgs, out *Void) os.Error {
	s.dev.linearCombination(unsafe.Pointer(in.A), unsafe.Pointer(in.B), in.WeightA, in.WeightB, in.N)
	return nil
}

func (s *DeviceServer) AddConstant(in *AddConstantArgs, out *Void) os.Error {
	s.dev.addConstant(unsafe.Pointer(in.A), in.Cnst, in.N)
	return nil
}

func (s *DeviceServer) Normalize(in *NormalizeArgs, out *Void) os.Error {
	s.dev.normalize(unsafe.Pointer(in.M), in.N)
	return nil
}

func (s *DeviceServer) NormalizeMap(in *NormalizeMapArgs, out *Void) os.Error {
	s.dev.normalizeMap(unsafe.Pointer(in.M), unsafe.Pointer(in.NormMap), in.N)
	return nil
}

func (s *DeviceServer) DeltaM(in *DeltaMArgs, out *Void) os.Error {
	s.dev.deltaM(unsafe.Pointer(in.M), unsafe.Pointer(in.H), in.Alpha, in.DtGilbert, in.N)
	return nil
}

func (s *DeviceServer) SemianalStep(in *SemianalStepArgs, out *Void) os.Error {
	s.dev.semianalStep(unsafe.Pointer(in.M), unsafe.Pointer(in.H), in.Dt, in.Alpha, in.Order, in.N)
	return nil
}

func (s *DeviceServer) KernelMul(in *KernelMulArgs, out *Void) os.Error {
	s.dev.kernelMul(unsafe.Pointer(in.Mx), unsafe.Pointer(in.My), unsafe.Pointer(in.Mz), unsafe.Pointer(in.Kxx), unsafe.Pointer(in.Kyy), unsafe.Pointer(in.Kzz), unsafe.Pointer(in.Kyz), unsafe.Pointer(in.Kxz), unsafe.Pointer(in.Kxy), in.Kerneltype, in.NRealNumbers)
	return nil
}


func (s *DeviceServer) CopyPadded(in *CopyPaddedArgs, out *Void) os.Error {
	s.dev.copyPadded(unsafe.Pointer(in.Source), unsafe.Pointer(in.Dest), in.SourceSize, in.DestSize, in.Direction)
	return nil
}


func (s *DeviceServer) NewFFTPlan(in *NewFFTPlanArgs, out *Ptr) os.Error {
	out.Value = uintptr(s.dev.newFFTPlan(in.DataSize, in.LogicSize))
	return nil
}


func (s *DeviceServer) FFT(in *FFTArgs, out *Void) os.Error {
	s.dev.fft(unsafe.Pointer(in.Plan), unsafe.Pointer(in.In), unsafe.Pointer(in.Out), in.Direction)
	return nil
}


func (s *DeviceServer) NewArray(in *Int, out *Ptr) os.Error {
	out.Value = uintptr(s.dev.newArray(in.Value))
	Debugvv("NewArray(", in, ") :", out)
	return nil
}
//
//
// func (s *DeviceServer) memcpy(source, dest unsafe.Pointer, nFloats, direction int) {
// 	C.memcpy_gpu_dir((*C.float)(unsafe.Pointer(source)), (*C.float)(dest), C.int(nFloats), C.int(direction))
// }
//
//
// func (s *DeviceServer) arrayOffset(array unsafe.Pointer, index int) unsafe.Pointer {
// 	return unsafe.Pointer(C.gpu_array_offset((*C.float)(array), C.int(index)))
// }
//
// func (s *DeviceServer) Stride() int {
// 	return int(C.gpu_stride_float())
// }
//
// func (s *DeviceServer) overrideStride(nFloats int) {
// 	C.gpu_override_stride(C.int(nFloats))
// }
//
func (s *DeviceServer) Zero(in *ZeroArgs, out *Void) os.Error {
  Debugvv("Zero", in)
	s.dev.zero(unsafe.Pointer(in.Data), in.NFloats)
	return nil
}
//

// func (s *DeviceServer) PrintProperties() {
//  C.gpu_print_properties_stdout()
// }

// func TimerPrintDetail(){
//   C.timer_printdetail()
// }



func (s *DeviceServer) String() string {
	return "Simulation server on " + s.port
}
