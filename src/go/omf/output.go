//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package omf

import (
  "io"
  "bufio"
  "fmt"
  "tensor"
)

const(

)

func (c *OmfCodec) Encode(out_ io.Writer, t tensor.Interface, metadata map[string]string) {
  out := bufio.NewWriter(out_)

  fmt.Println(out, FILE_HEADER)
  
}

func hdr(out io.Writer, 