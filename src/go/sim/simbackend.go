package sim

// This file implements the methods
// for backend (hardware) selection

import "strings"

// Select the CPU as backend
func (s *Sim) Cpu() {
	s.backend = CPU
	s.invalidate()
}

// Select the GPU as backend
func (s *Sim) Gpu() {
	s.backend = GPU
	s.invalidate()
}

// DEBUG select a remote device.
// Only useful to test the connection.
// Normally, you would use more than one remote device in a cluster.
func (s *Sim) Remote(transport string, serverAddress string, serverPort int) {
	transport = strings.ToLower(transport)
	s.backend = &Backend{NewRemoteDevice(transport, serverAddress, serverPort), false}
	s.invalidate()
}
