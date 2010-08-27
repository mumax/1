package sim

// This file implements the methods
// for backend (hardware) selection

import "strings"

// Select the CPU as backend
func (s *Sim) Cpu() {
	s.backend = CPU
	Debugv("Selected CPU backend")
	s.invalidate()
}

// Select the GPU as backend
func (s *Sim) Gpu() {
	s.backend = GPU
	Debugv("Selected CPU backend")
	s.invalidate()
}

// DEBUG select a remote device.
// Only useful to test the connection.
// Normally, you would use more than one remote device in a cluster.
func (s *Sim) Remote(transport string, serverAddress string, serverPort int) {
	Debugv("Selected remote backend:", transport, serverAddress, ":", serverPort)
	transport = strings.ToLower(transport)
	s.backend = &Backend{NewRemoteDevice(transport, serverAddress, serverPort), false}
	s.invalidate()
}
