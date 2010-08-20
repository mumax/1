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

// TODO: clusers etc.
