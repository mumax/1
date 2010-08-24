package sim

import (
  "unsafe"
  "rpc"
  "os"
)

// temp debug func
func Main(){
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
  serverPort string
}

// func NewRemoteBackend(serverAddress, serverPort string) Backend{
//   return &Backend{NewRemoteDevice(serverAddress, serverPort), false}
// }

func NewRemoteDevice(serverAddress, serverPort string) *RemoteDevice{
  d := new(RemoteDevice)
  d.serverAddress = serverAddress
  d.serverPort = serverPort
  var err os.Error
  Debugv("Dial tcp " + d.serverAddress + d.serverPort)
  d.Client, err = rpc.DialHTTP("tcp", d.serverAddress + d.serverPort)
  if err != nil {
    panic(err)
  }
  return d
}

func (d *RemoteDevice) init() {

}

type Void struct{}

type UIntPtr uintptr

type AddArgs struct{
  A, B uintptr
  N int
}

func (d *RemoteDevice) add(a, b unsafe.Pointer, N int) {
  args := &AddArgs{uintptr(a), uintptr(b),N}
  var reply int
  err := d.Client.Call("DeviceServer.Add", args, &reply)
  if err != nil {
    panic(err)
  }
}

