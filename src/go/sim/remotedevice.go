package sim

// TODO: it would be nice to reduce the number of funcs here
// DONE: Memcpy(DIRECTION)
// Normalize(map=nil)
// ...

import (
	"unsafe"
	"rpc"
	"os"
	"fmt"
)

// A RemoteDevice gives access to a Device (GPU, CPU, ...)
// on a remote server. Since RemoteDevice satisfies the Device
// interface, it can be used just like any other local Device.
//
type RemoteDevice struct {
	*rpc.Client
	transport, serverAddress string
	serverPort               int
}


// E.g.: NewRemoteDevice("tcp", "127.0.0.1", 2527)
func NewRemoteDevice(transport, serverAddress string, serverPort int) *RemoteDevice {
	d := new(RemoteDevice)
	d.transport = transport
	d.serverAddress = serverAddress
	d.serverPort = serverPort
	url := d.serverAddress + ":" + fmt.Sprint(d.serverPort)
	var err os.Error
	d.Client, err = rpc.DialHTTP(d.transport, url)
	if err != nil {
		panic(err)
	}
	Debugv("Connected to " + d.serverAddress + fmt.Sprint(d.serverPort))
	return d
}

func (d *RemoteDevice) init() {
	args := &Void{0}
	reply := &Void{0}
	err := d.Client.Call("DeviceWrapper.Init", args, reply)
	if err != nil {
		panic(err)
	}
}

type Void struct {
	Dummy int
}


type AddArgs struct {
	A, B uintptr
	N    int
}

func (d *RemoteDevice) add(a, b unsafe.Pointer, N int) {
	args := &AddArgs{uintptr(a), uintptr(b), N}
	reply := &Void{0}
	err := d.Client.Call("DeviceWrapper.Add", args, reply)
	if err != nil {
		panic(err)
	}
}

//
type LinearCombinationArgs struct {
	A, B             uintptr
	WeightA, WeightB float
	N                int
}

func (d *RemoteDevice) linearCombination(a, b unsafe.Pointer, weightA, weightB float, N int) {
	args := &LinearCombinationArgs{uintptr(a), uintptr(b), weightA, weightB, N}
	reply := &Void{0}
	err := d.Client.Call("DeviceWrapper.LinearCombination", args, reply)
	if err != nil {
		panic(err)
	}
}

type AddConstantArgs struct {
	A    uintptr
	Cnst float
	N    int
}

func (d *RemoteDevice) addConstant(a unsafe.Pointer, cnst float, N int) {
	args := &AddConstantArgs{uintptr(a), cnst, N}
	reply := &Void{0}
	err := d.Client.Call("DeviceWrapper.AddConstant", args, reply)
	if err != nil {
		panic(err)
	}
}

type NormalizeArgs struct {
	M uintptr
	N int
}

func (d *RemoteDevice) normalize(m unsafe.Pointer, N int) {
	args := &NormalizeArgs{uintptr(m), N}
	reply := &Void{0}
	err := d.Client.Call("DeviceWrapper.Normalize", args, reply)
	if err != nil {
		panic(err)
	}
}

type NormalizeMapArgs struct {
	M, NormMap uintptr
	N          int
}

func (d *RemoteDevice) normalizeMap(m, normMap unsafe.Pointer, N int) {
	args := &NormalizeMapArgs{uintptr(m), uintptr(normMap), N}
	reply := &Void{0}
	err := d.Client.Call("DeviceWrapper.NormalizeMap", args, reply)
	if err != nil {
		panic(err)
	}
}

type DeltaMArgs struct {
	M, H             uintptr
	Alpha, DtGilbert float
	N                int
}

func (d *RemoteDevice) deltaM(m, h unsafe.Pointer, alpha, dtGilbert float, N int) {
	args := &DeltaMArgs{uintptr(m), uintptr(h), alpha, dtGilbert, N}
	reply := &Void{0}
	err := d.Client.Call("DeviceWrapper.DeltaM", args, reply)
	if err != nil {
		panic(err)
	}
}

type SemianalStepArgs struct {
	M, H      uintptr
	Dt, Alpha float
	Order, N  int
}

func (d *RemoteDevice) semianalStep(m, h unsafe.Pointer, dt, alpha float, order, N int) {
	var args = &SemianalStepArgs{uintptr(m), uintptr(h), dt, alpha, order, N}
	reply := &Void{0}
	err := d.Client.Call("DeviceWrapper.SemiAnalStep", &args, reply)
	if err != nil {
		panic(err)
	}
}

type Int struct {
	Value int
}

func (d *RemoteDevice) newArray(nFloats int) unsafe.Pointer {
	var args = &Int{nFloats}
	reply := &Ptr{0}
	err := d.Client.Call("DeviceWrapper.NewArray", args, reply)
	if err != nil {
		panic(err)
	}
	Debugvv("newArray(", args, "): ", reply)
	return unsafe.Pointer(reply.Value) // WARNING  unsafe.Pointer(reply) is not a compilation error but is wrong!
}


type MemcpyArgs struct {
	Source, Dest       uintptr
	NFloats, Direction int
}

func (d *RemoteDevice) memcpy(source, dest unsafe.Pointer, nFloats, direction int) {

	args := &MemcpyArgs{uintptr(unsafe.Pointer(source)), uintptr(dest), nFloats, direction}
	reply := &Void{0}
	err := d.Client.Call("DeviceWrapper.Memcpy", args, reply)
	if err != nil {
		panic(err)
	}

}


type ZeroArgs struct {
	Data    uintptr
	NFloats int
}

func (d *RemoteDevice) zero(data unsafe.Pointer, nFloats int) {
	args := &ZeroArgs{uintptr(data), nFloats}
	reply := &Void{0}
	Debugvv("zero(", args, ")")
	err := d.Client.Call("DeviceWrapper.Zero", args, reply)
	if err != nil {
		panic(err)
	}
}

type KernelMulArgs struct {
	Mx, My, Mz, Kxx, Kyy, Kzz, Kyz, Kxz, Kxy uintptr
	Kerneltype, NRealNumbers                 int
}

func (d *RemoteDevice) kernelMul(mx, my, mz, kxx, kyy, kzz, kyz, kxz, kxy unsafe.Pointer, kerneltype, nRealNumbers int) {
	args := &KernelMulArgs{uintptr(mx), uintptr(my), uintptr(mz), uintptr(kxx), uintptr(kyy), uintptr(kzz), uintptr(kyz), uintptr(kxz), uintptr(kxy), kerneltype, nRealNumbers}
	reply := &Void{0}
	err := d.Client.Call("DeviceWrapper.KernelMul", args, reply)
	if err != nil {
		panic(err)
	}
}

type NewFFTPlanArgs struct {
	DataSize, LogicSize []int
}

func (d *RemoteDevice) newFFTPlan(dataSize, logicSize []int) unsafe.Pointer {
	args := &NewFFTPlanArgs{dataSize, logicSize}
	reply := &Ptr{0}
	err := d.Client.Call("DeviceWrapper.NewFFTPlan", args, reply)
	if err != nil {
		panic(err)
	}
	return unsafe.Pointer(reply.Value)
}

type FFTArgs struct {
	Plan, In, Out uintptr
	Direction     int
}

type Ptr struct {
	Value uintptr
}

func (d *RemoteDevice) fft(plan unsafe.Pointer, in, out unsafe.Pointer, direction int) {
	args := &FFTArgs{uintptr(plan), uintptr(in), uintptr(out), direction}
	reply := &Void{0}
	err := d.Client.Call("DeviceWrapper.FFT", args, reply)
	if err != nil {
		panic(err)
	}
}

type CopyPaddedArgs struct {
	Source, Dest         uintptr
	SourceSize, DestSize []int
	Direction            int
}

func (d *RemoteDevice) copyPadded(source, dest unsafe.Pointer, sourceSize, destSize []int, direction int) {
	args := &CopyPaddedArgs{uintptr(source), uintptr(dest), sourceSize, destSize, direction}
	reply := &Void{0}
	err := d.Client.Call("DeviceWrapper.CopyPadded", args, reply)
	if err != nil {
		panic(err)
	}
}


type ArrayOffsetArgs struct {
	Array uintptr
	Index int
}

func (d *RemoteDevice) arrayOffset(array unsafe.Pointer, index int) unsafe.Pointer {
	args := &ArrayOffsetArgs{uintptr(array), index}
	reply := &Ptr{0}
	err := d.Client.Call("DeviceWrapper.ArrayOffset", args, reply)
	if err != nil {
		panic(err)
	}
	return unsafe.Pointer(reply.Value)
}


func (d *RemoteDevice) Stride() int {
	reply := &Int{0}
	err := d.Client.Call("DeviceWrapper.Stride", Void{}, reply)
	if err != nil {
		panic(err)
	}
	return reply.Value
}

func (d *RemoteDevice) overrideStride(nFloats int) {
	reply := &Void{0}
	err := d.Client.Call("DeviceWrapper.OverrideStride", &Int{nFloats}, reply)
	if err != nil {
		panic(err)
	}
}

func (d *RemoteDevice) PrintProperties() {
	fmt.Println(d.String())
}


func (d *RemoteDevice) TimerPrintDetail() {
	//C.timer_printdetail()
	fmt.Println("meh...")
}

func (d *RemoteDevice) String() string {
	return "RemoteDevice: " + d.serverAddress + ":" + fmt.Sprint(d.serverPort)
}