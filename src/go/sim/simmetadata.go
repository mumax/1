//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

// This file implements the methods for setting
// simulation metadata

// Sets a custom metadata key-value pair.
// This pair will be added to the header of all
// output, which can be handy when analyzing data.
// I.e., to tell you (or an analysis script)
// the value of a parameter that is otherwise unclear.
// E.g.: # initial_state: magnetzation_up
func (s *Sim) Metadata(key, value string) {
	s.metadata[key] = value
}
