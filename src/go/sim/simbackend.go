package sim

// This file implements the methods
// for backend (hardware) selection

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
func (s *Sim) Remote(serverAddress string, serverPort int) {
	s.backend = &Backend{NewRemoteDevice(serverAddress, serverPort), false}
	s.invalidate()
}
