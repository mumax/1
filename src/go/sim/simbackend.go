//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

// OBSOLETE: backend selection is not done by input file anymore,
// but as flag to the main program.

// This file implements the methods
// for backend (hardware) selection

// import "strings"

// Select the CPU as backend
// func (s *Sim) Cpu() {
// 	s.backend = CPU
// 	s.Println("Selected CPU backend")
// 	s.invalidate()
// }

// Select the GPU as backend
// func (s *Sim) Gpu() {
// 	s.backend = GPU
// 	s.Println("Selected GPU backend")
// 	s.backend.PrintProperties()
// 	s.invalidate()
// }

// DEBUG select a remote device.
// Only useful to test the connection.
// Normally, you would use more than one remote device in a cluster.
// func (s *Sim) Remote(transport string, serverAddress string, serverPort int) {
// 	Debugv("Selected remote backend:", transport, serverAddress, ":", serverPort)
// 	transport = strings.ToLower(transport)
// 	s.backend = NewBackend(NewRemoteDevice(transport, serverAddress, serverPort))
// 	s.invalidate()
// }
