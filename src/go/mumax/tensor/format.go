//  This file is part of MuMax, a high-performance micromagnetic simulator
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package tensor

// This file provides human-readable tensor formatting

import (
	"mumax/common"
	"fmt"
	"io"
	"bufio"
)

// Writes the tensor data as ascii.
// Includes some newlines to make it human-readable
func Format(out_ io.Writer, t Interface) {
	out := bufio.NewWriter(out_)
	defer out.Flush()

	for i := NewIterator(t); i.HasNext(); i.Next() {
		if _, err := fmt.Fprint(out, i.Get(), "\t"); err != nil {
			common.CheckErr(err, common.ERR_IO)
		}

		for j := 0; j < Rank(t); j++ {
			newline := true
			for k := j; k < Rank(t); k++ {
				if i.Index()[k] != t.Size()[k]-1 {
					newline = false
				}
			}
			if newline {
				if _, err := fmt.Fprint(out, "\n"); err != nil {
					common.CheckErr(err, common.ERR_IO)
				}
			}
		}
	}
}
