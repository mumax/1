//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package tensor

import (
	"io"
)

type Codec interface {
	Encode(out io.Writer, t Interface, metadata map[string]string)
	Decode(in io.Reader) (*T, map[string]string)
}

type TensorCodec struct {
	format uint
}

func (c *TensorCodec) Init() {

}

func NewTensorCodec() *TensorCodec {
	c := new(TensorCodec)
	c.Init()
	return c
}

func (c *TensorCodec) Encode(out io.Writer, t Interface, metadata map[string]string) {
	WriteMetaTensorBinary(out, t, metadata)
}

func (c *TensorCodec) Decode(in io.Reader) (tensor *T, metadata map[string]string) {
	tensor, metadata = ReadMeta(in)
	return
}
