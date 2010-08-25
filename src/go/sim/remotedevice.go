package sim

// TODO: it would be nice to reduce the number of funcs here
// DONE: Memcpy(DIRECTION)
// Normalize(map=nil)
// ...

import (
	"unsafe"
	"rpc"
	"os"
)

// temp debug func
func Main() {
	dev := NewRemoteDevice("127.0.0.1", ":2527")
	dev.add(nil, nil, 100)
}

// A RemoteDevice gives access to a Device (GPU, CPU, ...)
// on a remote server. Since RemoteDevice satisfies the Device
// interface, it can be used just like any other local Device.
//
type RemoteDevice struct {
	*rpc.Client
	serverAddress string
	serverPort    string
}

// func NewRemoteBackend(serverAddress, serverPort string) Backend{
//   return &Backend{NewRemoteDevice(serverAddress, serverPort), false}
// }

func NewRemoteDevice(serverAddress, serverPort string) *RemoteDevice {
	d := new(RemoteDevice)
	d.serverAddress = serverAddress
	d.serverPort = serverPort
	var err os.Error
	d.Client, err = rpc.DialHTTP("tcp", d.serverAddress+d.serverPort) //TODO: UDP
	if err != nil {
		panic(err)
	}
	Debugv("Connected to " + d.serverAddress + d.serverPort)
	return d
}

func (d *RemoteDevice) init() {

}

type Void struct{}


type AddArgs struct {
	A, B uintptr
	N    int
}

func (d *RemoteDevice) add(a, b unsafe.Pointer, N int) {
	args := &AddArgs{uintptr(a), uintptr(b), N}
	var reply int
	err := d.Client.Call("DeviceServer.Add", args, &reply)
	if err != nil {
		panic(err)
	}
}


type LinearCombinationArgs struct {
	A, B             uintptr
	WeightA, WeightB float
	N                int
}

func (d *RemoteDevice) linearCombination(a, b unsafe.Pointer, weightA, weightB float, N int) {
	args := &LinearCombinationArgs{uintptr(a), uintptr(b), weightA, weightB, N}
	var reply int
	err := d.Client.Call("DeviceServer.LinearCombination", args, &reply)
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
	var reply int
	err := d.Client.Call("DeviceServer.AddConstant", args, &reply)
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
	var reply int
	err := d.Client.Call("DeviceServer.Normalize", args, &reply)
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
	var reply int
	err := d.Client.Call("DeviceServer.NormalizeMap", args, &reply)
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
	var reply int
	err := d.Client.Call("DeviceServer.DeltaM", args, &reply)
	if err != nil {
		panic(err)
	}
}

type Int int

func (d *RemoteDevice) newArray(nFloats int) unsafe.Pointer {
  var args = Int(nFloats)
  var reply uintptr
  err := d.Client.Call("DeviceServer.NewArray", &args, &reply)
  if err != nil {
    panic(err)
  }
  return unsafe.Pointer(reply)
}

type MemcpyArgs struct{
  Source, Dest uintptr
  NFloats, direction int
}

func (d *RemoteDevice) memcpy(source, dest unsafe.Pointer, nFloats, direction int) {
  args := &MemcpyArgs{uintptr(unsafe.Pointer(source)), uintptr(dest), nFloats, direction}
  var reply int
  err := d.Client.Call("DeviceServer.MemcpyTo", args, &reply)
  if err != nil {
    panic(err)
  }
}

