//  This file is part of MuMax, a high-performance micromagnetic simulator
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
func (s *Sim) Desc(key, value string) {
	if s.tabwriter == nil { s.initTabWriter()}
	// We separately add the desc tag to the omf and odt output.
	// The odt output will only contain the manually added descriptions,
	// which are supposedly constant and applicable to the entire table.
	// The omf output will additionally contain automatically added
	// tags that may change over time like, e.g., the applied field.
	s.tabwriter.AddDesc(key, value)
	s.desc[key] = value
}
