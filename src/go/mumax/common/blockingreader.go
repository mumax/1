//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package common

import (
	"io"
	"os"
)

// Blocks until all requested bytes are read.
type BlockingReader struct {
	In io.Reader
}


func (r *BlockingReader) Read(p []byte) (n int, err os.Error) {
	n, err = r.In.Read(p)
	if err != nil {
		if err == os.EOF {
			return
		} else {
			panic(IOErr(err.String()))
		}
	}
	if n < len(p) {
		_, err = r.Read(p[n:])
	}
	if err != nil {
		if err == os.EOF {
			return
		} else {
			panic(IOErr(err.String()))
		}
	}
	n = len(p)
	return
}


func NewBlockingReader(in io.Reader) *BlockingReader {
	return &BlockingReader{in}
}
