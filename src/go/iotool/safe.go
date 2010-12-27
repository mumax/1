//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package iotool

import (
  "io"
  "os"
)

// Blocks until all requested bytes are read.
// Never returns an error but panics instead
type SafeReader struct{
  In io.Reader
}


func (r *SafeReader) Read(p []byte) (n int, err os.Error){
  n, err = r.In.Read(p)
  if err != nil{panic(err)}
  if n < len(p){
    r.Read(p[n-1:])
  }
  n = len(p)
  return
}


func NewSafeReader(in io.Reader) *SafeReader{
  return &SafeReader{in}
}