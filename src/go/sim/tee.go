//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

// Plumbing with pipes.

import (
	"io"
	//"os"
	//"fmt"
)

// Reads from in and passes data through to out.
// Typically ran inside a separate goroutine.
func Pipe(in io.Reader, out io.Writer) {
	buf := [4096]byte{}[:]
	for {
		//fmt.Fprint(os.Stderr, "Pipe waiting for read ") //debug
		n, err := in.Read(buf)
		//fmt.Fprintln(os.Stderr, "OK") //debug
		if err != nil {
			//fmt.Fprintln(os.Stderr, "Pipe: ", err) //debug
			if closer := out.(io.Closer); closer != nil {
				closer.Close()
			}
			return
		}
		//fmt.Fprintln(os.Stderr, "Pipe[", string(buf[:n]), "]") //debug
		out.Write(buf[:n])
	}
}
