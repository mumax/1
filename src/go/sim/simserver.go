//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

// import (
// 	"unsafe"
// 	"rpc"
// 	"net"
// 	"http"
// 	"os"
// 	"fmt"
// 	"strings"
// )
// 
// // Runs a simulation server over the network.
// // On one end, a "RemoteDevice" interface is used to access the remote server.
// // On the other end of the connection, a DeviceServer accepts remote function
// // calls and does the number-crunching.
// //
// // This code is not yet ready for prime-time.
// // It is mainly intended for playing with networked devices.
// //
// type DeviceServer struct {
// 	*DeviceWrapper
// 	transport string // udp or tcp
// 	port      string
// }
// 
// func NewDeviceServer(device string, transport string, port int) *DeviceServer {
// 	var d Device
// 	switch strings.ToLower(device) {
// 	default:
// 		panic("Unknown device: " + device)
// 	case "cpu":
// 		d = CPU.Device
// 	case "gpu":
// 		d = GPU.Device
// 	}
// 	return &DeviceServer{&DeviceWrapper{d}, transport, fmt.Sprintf(":%d", port)}
// }
// 
// func (server *DeviceServer) Listen() {
// 	rpc.Register(server.DeviceWrapper)
// 	rpc.HandleHTTP()
// 	listener, err := net.Listen(server.transport, server.port)
// 	if err != nil {
// 		panic(err)
// 	}
// 	Debug("Listening on "+server.transport+"port", server.port)
// 	http.Serve(listener, nil)
// }
// 
// type DeviceWrapper struct {
// 	dev Device // We do not embed to avoid the dev methods to be exported by rpc
// }
// 
// func (s *DeviceWrapper) Init(in, out *Void) os.Error {
// 	Debugvv("Init()")
// 	s.dev.init()
// 	return nil
// }
// 
// func (s *DeviceWrapper) Add(in *AddArgs, out *Void) os.Error {
// 	s.dev.add(unsafe.Pointer(in.A), unsafe.Pointer(in.B), in.N)
// 	return nil
// }
// 
// func (s *DeviceWrapper) LinearCombination(in *LinearCombinationArgs, out *Void) os.Error {
// 	s.dev.linearCombination(unsafe.Pointer(in.A), unsafe.Pointer(in.B), in.WeightA, in.WeightB, in.N)
// 	return nil
// }
// 
// func (s *DeviceWrapper) AddConstant(in *AddConstantArgs, out *Void) os.Error {
// 	s.dev.addConstant(unsafe.Pointer(in.A), in.Cnst, in.N)
// 	return nil
// }
// 
// func (s *DeviceWrapper) Normalize(in *NormalizeArgs, out *Void) os.Error {
// 	s.dev.normalize(unsafe.Pointer(in.M), in.N)
// 	return nil
// }
// 
// func (s *DeviceWrapper) NormalizeMap(in *NormalizeMapArgs, out *Void) os.Error {
// 	s.dev.normalizeMap(unsafe.Pointer(in.M), unsafe.Pointer(in.NormMap), in.N)
// 	return nil
// }
// 
// func (s *DeviceWrapper) DeltaM(in *DeltaMArgs, out *Void) os.Error {
// 	s.dev.deltaM(unsafe.Pointer(in.M), unsafe.Pointer(in.H), in.Alpha, in.DtGilbert, in.N)
// 	return nil
// }
// 
// func (s *DeviceWrapper) SemianalStep(in *SemianalStepArgs, out *Void) os.Error {
// 	s.dev.semianalStep(unsafe.Pointer(in.M), unsafe.Pointer(in.H), in.Dt, in.Alpha, in.Order, in.N)
// 	return nil
// }
// 
// func (s *DeviceWrapper) KernelMul(in *KernelMulArgs, out *Void) os.Error {
// 	s.dev.kernelMul(unsafe.Pointer(in.Mx), unsafe.Pointer(in.My), unsafe.Pointer(in.Mz), unsafe.Pointer(in.Kxx), unsafe.Pointer(in.Kyy), unsafe.Pointer(in.Kzz), unsafe.Pointer(in.Kyz), unsafe.Pointer(in.Kxz), unsafe.Pointer(in.Kxy), in.Kerneltype, in.NRealNumbers)
// 	return nil
// }
// 
// 
// func (s *DeviceWrapper) CopyPadded(in *CopyPaddedArgs, out *Void) os.Error {
// 	s.dev.copyPadded(unsafe.Pointer(in.Source), unsafe.Pointer(in.Dest), in.SourceSize, in.DestSize, in.Direction)
// 	return nil
// }
// 
// 
// func (s *DeviceWrapper) NewFFTPlan(in *NewFFTPlanArgs, out *Ptr) os.Error {
// 	out.Value = unsafe.Pointer(s.dev.newFFTPlan(in.DataSize, in.LogicSize))
// 	return nil
// }
// 
// 
// func (s *DeviceWrapper) FFT(in *FFTArgs, out *Void) os.Error {
// 	s.dev.fft(unsafe.Pointer(in.Plan), unsafe.Pointer(in.In), unsafe.Pointer(in.Out), in.Direction)
// 	return nil
// }
// 
// 
// func (s *DeviceWrapper) NewArray(in *Int, out *Ptr) os.Error {
// 	out.Value = unsafe.Pointer(s.dev.newArray(in.Value))
// 	Debugvv("NewArray(", in, ") :", out)
// 	return nil
// }
// 
// 
// func (s *DeviceWrapper) Memcpy(in *MemcpyArgs, out *Void) os.Error {
// 
// 	switch in.Direction {
// 	default:
// 		panic(fmt.Sprintf("Unknown memcpy direction: ", in.Direction))
// 	case CPY_TO:
// 	case CPY_ON:
// 		s.dev.memcpy(unsafe.Pointer(in.Source), unsafe.Pointer(in.Dest), in.NFloats, in.Direction)
// 	case CPY_FROM:
// 	}
// 
// 	return nil
// }
// 
// 
// func (s *DeviceWrapper) ArrayOffset(in *ArrayOffsetArgs, out *Ptr) os.Error {
// 	out.Value = unsafe.Pointer(s.dev.arrayOffset(unsafe.Pointer(in.Array), in.Index))
// 	Debugvv("ArrayOffset(", in, "):", out)
// 	return nil
// }
// 
// func (s *DeviceWrapper) Stride(in *Void, out *Int) os.Error {
// 	out.Value = s.dev.Stride()
// 	return nil
// }
// //
// // func (s *DeviceWrapper) overrideStride(nFloats int) {
// // 	C.gpu_override_stride(C.int(nFloats))
// // }
// //
// func (s *DeviceWrapper) Zero(in *ZeroArgs, out *Void) os.Error {
// 	Debugvv("Zero(", in, ")")
// 	s.dev.zero(unsafe.Pointer(in.Data), in.NFloats)
// 	return nil
// }
// //
// 
// // func (s *DeviceWrapper) PrintProperties() {
// //  C.gpu_print_properties_stdout()
// // }
// 
// // func TimerPrintDetail(){
// //   C.timer_printdetail()
// // }
// 
// 
// // func (s *DeviceWrapper) String() string {
// // 	return "Simulation server on " + s.port
// // }
