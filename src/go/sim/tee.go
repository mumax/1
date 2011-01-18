//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

// Plumbing with pipes.

import (
	"io"
	"os"
)

// Reads from in and passes data through to out.
// Typically ran inside a separate goroutine.
// TODO: close out when in is closed
func Pipe(in io.Reader, out io.Writer) {
	buf := [512]byte{}[:]
	for {
		n, err := in.Read(buf)
		if err != nil {
			return
		}
		os.Stdout.Write(buf[:n])
	}
}
