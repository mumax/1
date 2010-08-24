package main

import (
	. "sim"
	"rpc"
	"net"
	"http"
	"os"
	"fmt"
)

type DeviceServer struct {
	dev *Device
	port string
}

func main() {
  Verbosity = 3
  
	server := &DeviceServer{&CPU.Device, ":2527"}
	rpc.Register(server)
	rpc.HandleHTTP()
	listener, err := net.Listen("tcp", server.port)
	if err != nil {
		panic(err)
	}
	fmt.Println("Listening on port", server.port)
	http.Serve(listener, nil)
}


func (s *DeviceServer) Add(in *AddArgs, out *Void) os.Error{
  Debugvv("Add")
  return nil
}
