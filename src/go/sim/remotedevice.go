//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

// TODO: it would be nice to reduce the number of funcs here
// DONE: Memcpy(DIRECTION)
// Normalize(map=nil)
// ...

// import (
// 	"unsafe"
// 	"rpc"
// 	"os"
// 	"fmt"
// )
// 
// // A RemoteDevice gives access to a Device (GPU, CPU, ...)
// // on a remote server. Since RemoteDevice satisfies the Device
// // interface, it can be used just like any other local Device.
// //
// type RemoteDevice struct {
// 	*rpc.Client
// 	transport, serverAddress string
// 	serverPort               int
// }
// 
// 
// // E.g.: NewRemoteDevice("tcp", "127.0.0.1", 2527)
// func NewRemoteDevice(transport, serverAddress string, serverPort int) *RemoteDevice {
// 	d := new(RemoteDevice)
// 	d.transport = transport
// 	d.serverAddress = serverAddress
// 	d.serverPort = serverPort
// 	url := d.serverAddress + ":" + fmt.Sprint(d.serverPort)
// 	var err os.Error
// 	d.Client, err = rpc.DialHTTP(d.transport, url)
// 	if err != nil {
// 		panic(err)
// 	}
// 	Debugv("Connected to " + d.serverAddress + fmt.Sprint(d.serverPort))
// 	return d
// }
// 
// func (d *RemoteDevice) init() {
// 	args := &Void{0}
// 	reply := &Void{0}
// 	err := d.Client.Call("DeviceWrapper.Init", args, reply)
// 	if err != nil {
// 		panic(err)
// 	}
// }
// 
// type Void struct {
// 	Dummy int
// }
// 
// 
// type AddArgs struct {
// 	A, B uintptr
// 	N    int
// }
// 
// func (d *RemoteDevice) add(a, b uintptr, N int) {
// 	args := &AddArgs{unsafe.Pointer(a), unsafe.Pointer(b), N}
// 	reply := &Void{0}
// 	err := d.Client.Call("DeviceWrapper.Add", args, reply)
// 	if err != nil {
// 		panic(err)
// 	}
// }
// 
// //
// type LinearCombinationArgs struct {
// 	A, B             uintptr
// 	WeightA, WeightB float
// 	N                int
// }
// 
// func (d *RemoteDevice) linearCombination(a, b uintptr, weightA, weightB float, N int) {
// 	args := &LinearCombinationArgs{unsafe.Pointer(a), unsafe.Pointer(b), weightA, weightB, N}
// 	reply := &Void{0}
// 	err := d.Client.Call("DeviceWrapper.LinearCombination", args, reply)
// 	if err != nil {
// 		panic(err)
// 	}
// }
// 
// type AddConstantArgs struct {
// 	A    uintptr
// 	Cnst float
// 	N    int
// }
// 
// func (d *RemoteDevice) addConstant(a uintptr, cnst float, N int) {
// 	args := &AddConstantArgs{unsafe.Pointer(a), cnst, N}
// 	reply := &Void{0}
// 	err := d.Client.Call("DeviceWrapper.AddConstant", args, reply)
// 	if err != nil {
// 		panic(err)
// 	}
// }
// 
// type NormalizeArgs struct {
// 	M uintptr
// 	N int
// }
// 
// func (d *RemoteDevice) normalize(m uintptr, N int) {
// 	args := &NormalizeArgs{unsafe.Pointer(m), N}
// 	reply := &Void{0}
// 	err := d.Client.Call("DeviceWrapper.Normalize", args, reply)
// 	if err != nil {
// 		panic(err)
// 	}
// }
// 
// type NormalizeMapArgs struct {
// 	M, NormMap uintptr
// 	N          int
// }
// 
// func (d *RemoteDevice) normalizeMap(m, normMap uintptr, N int) {
// 	args := &NormalizeMapArgs{unsafe.Pointer(m), unsafe.Pointer(normMap), N}
// 	reply := &Void{0}
// 	err := d.Client.Call("DeviceWrapper.NormalizeMap", args, reply)
// 	if err != nil {
// 		panic(err)
// 	}
// }
// 
// type DeltaMArgs struct {
// 	M, H             uintptr
// 	Alpha, DtGilbert float
// 	N                int
// }
// 
// func (d *RemoteDevice) deltaM(m, h uintptr, alpha, dtGilbert float, N int) {
// 	args := &DeltaMArgs{unsafe.Pointer(m), unsafe.Pointer(h), alpha, dtGilbert, N}
// 	reply := &Void{0}
// 	err := d.Client.Call("DeviceWrapper.DeltaM", args, reply)
// 	if err != nil {
// 		panic(err)
// 	}
// }
// 
// type SemianalStepArgs struct {
// 	M, H      uintptr
// 	Dt, Alpha float
// 	Order, N  int
// }
// 
// func (d *RemoteDevice) semianalStep(m, h uintptr, dt, alpha float, order, N int) {
// 	var args = &SemianalStepArgs{unsafe.Pointer(m), unsafe.Pointer(h), dt, alpha, order, N}
// 	reply := &Void{0}
// 	err := d.Client.Call("DeviceWrapper.SemiAnalStep", &args, reply)
// 	if err != nil {
// 		panic(err)
// 	}
// }
// 
// type Int struct {
// 	Value int
// }
// 
// func (d *RemoteDevice) newArray(nFloats int) uintptr {
// 	var args = &Int{nFloats}
// 	reply := &Ptr{0}
// 	err := d.Client.Call("DeviceWrapper.NewArray", args, reply)
// 	if err != nil {
// 		panic(err)
// 	}
// 	Debugvv("newArray(", args, "): ", reply)
// 	return unsafe.Pointer(reply.Value) // WARNING  unsafe.Pointer(reply) is not a compilation error but is wrong!
// }
// 
// 
// type MemcpyArgs struct {
// 	Source, Dest       uintptr
// 	NFloats, Direction int
// }
// 
// func (d *RemoteDevice) memcpy(source, dest uintptr, nFloats, direction int) {
// 
// 	args := &MemcpyArgs{unsafe.Pointer(unsafe.Pointer(source)), unsafe.Pointer(dest), nFloats, direction}
// 	reply := &Void{0}
// 	err := d.Client.Call("DeviceWrapper.Memcpy", args, reply)
// 	if err != nil {
// 		panic(err)
// 	}
// 
// }
// 
// 
// type ZeroArgs struct {
// 	Data    uintptr
// 	NFloats int
// }
// 
// func (d *RemoteDevice) zero(data uintptr, nFloats int) {
// 	args := &ZeroArgs{unsafe.Pointer(data), nFloats}
// 	reply := &Void{0}
// 	Debugvv("zero(", args, ")")
// 	err := d.Client.Call("DeviceWrapper.Zero", args, reply)
// 	if err != nil {
// 		panic(err)
// 	}
// }
// 
// type KernelMulArgs struct {
// 	Mx, My, Mz, Kxx, Kyy, Kzz, Kyz, Kxz, Kxy uintptr
// 	Kerneltype, NRealNumbers                 int
// }
// 
// func (d *RemoteDevice) kernelMul(mx, my, mz, kxx, kyy, kzz, kyz, kxz, kxy uintptr, kerneltype, nRealNumbers int) {
// 	args := &KernelMulArgs{unsafe.Pointer(mx), unsafe.Pointer(my), unsafe.Pointer(mz), unsafe.Pointer(kxx), unsafe.Pointer(kyy), unsafe.Pointer(kzz), unsafe.Pointer(kyz), unsafe.Pointer(kxz), unsafe.Pointer(kxy), kerneltype, nRealNumbers}
// 	reply := &Void{0}
// 	err := d.Client.Call("DeviceWrapper.KernelMul", args, reply)
// 	if err != nil {
// 		panic(err)
// 	}
// }
// 
// type NewFFTPlanArgs struct {
// 	DataSize, LogicSize []int
// }
// 
// func (d *RemoteDevice) newFFTPlan(dataSize, logicSize []int) uintptr {
// 	args := &NewFFTPlanArgs{dataSize, logicSize}
// 	reply := &Ptr{0}
// 	err := d.Client.Call("DeviceWrapper.NewFFTPlan", args, reply)
// 	if err != nil {
// 		panic(err)
// 	}
// 	return unsafe.Pointer(reply.Value)
// }
// 
// type FFTArgs struct {
// 	Plan, In, Out uintptr
// 	Direction     int
// }
// 
// type Ptr struct {
// 	Value uintptr
// }
// 
// func (d *RemoteDevice) fft(plan uintptr, in, out uintptr, direction int) {
// 	args := &FFTArgs{unsafe.Pointer(plan), unsafe.Pointer(in), unsafe.Pointer(out), direction}
// 	reply := &Void{0}
// 	err := d.Client.Call("DeviceWrapper.FFT", args, reply)
// 	if err != nil {
// 		panic(err)
// 	}
// }
// 
// type CopyPaddedArgs struct {
// 	Source, Dest         uintptr
// 	SourceSize, DestSize []int
// 	Direction            int
// }
// 
// func (d *RemoteDevice) copyPadded(source, dest uintptr, sourceSize, destSize []int, direction int) {
// 	args := &CopyPaddedArgs{unsafe.Pointer(source), unsafe.Pointer(dest), sourceSize, destSize, direction}
// 	reply := &Void{0}
// 	err := d.Client.Call("DeviceWrapper.CopyPadded", args, reply)
// 	if err != nil {
// 		panic(err)
// 	}
// }
// 
// 
// type ArrayOffsetArgs struct {
// 	Array uintptr
// 	Index int
// }
// 
// func (d *RemoteDevice) arrayOffset(array uintptr, index int) uintptr {
// 	args := &ArrayOffsetArgs{unsafe.Pointer(array), index}
// 	reply := &Ptr{0}
// 	err := d.Client.Call("DeviceWrapper.ArrayOffset", args, reply)
// 	if err != nil {
// 		panic(err)
// 	}
// 	return unsafe.Pointer(reply.Value)
// }
// 
// 
// func (d *RemoteDevice) Stride() int {
// 	reply := &Int{0}
// 	err := d.Client.Call("DeviceWrapper.Stride", Void{}, reply)
// 	if err != nil {
// 		panic(err)
// 	}
// 	return reply.Value
// }
// 
// func (d *RemoteDevice) overrideStride(nFloats int) {
// 	reply := &Void{0}
// 	err := d.Client.Call("DeviceWrapper.OverrideStride", &Int{nFloats}, reply)
// 	if err != nil {
// 		panic(err)
// 	}
// }
// 
// func (d *RemoteDevice) UsedMem() uint64 {
// 	return 0 // meh
// }
// 
// func (d *RemoteDevice) PrintProperties() {
// 	fmt.Println(d.String())
// }
// 
// 
// func (d *RemoteDevice) TimerPrintDetail() {
// 	//C.timer_printdetail()
// 	fmt.Println("meh...")
// }
// 
// func (d *RemoteDevice) String() string {
// 	return "RemoteDevice: " + d.serverAddress + ":" + fmt.Sprint(d.serverPort)
// }
