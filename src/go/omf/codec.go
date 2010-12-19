//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.


/*
  This package implements a tensor.Codec that reads/writes data in
  OOMMF's .omf format.
*/
package omf

import (
  "io"
  "tensor"
)


type OmfCodec struct{
  format uint
}

func (c *OmfCodec) Init(){
  
}

func NewOmfCodec() *OmfCodec{
  c := new(OmfCodec)
  c.Init()
  return c
}

// Encode/Decode implemented in output.go/input.go
