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
)

const(

)


func (c *OmfCodec) Encode(out_ io.Writer, f Interface) {
  out := bufio.NewWriter(out_)
  defer out.Flush()
  
  format := "text"

  hdr(out, "OOMMF", "rectangular mesh v1.0")
  hdr(out, "Segment count", "1")
  hdr(out, "Begin", "Segment")
  
  hdr(out, "Begin", "Header")
  
  hdr(out, "Title", out_)
  hdr(out, "meshtype", "rectangular")
  
  hdr(out, "meshunit", f.MeshUnit())
  
  hdr(out, "xBase", 0)
  hdr(out, "yBase", 0)
  hdr(out, "zBase", 0)

  cellsize := f.CellSize()
  hdr(out, "xStepSize", cellsize[X])
  hdr(out, "xStepSize", cellsize[Y])
  hdr(out, "xStepSize", cellsize[Z])


  hdr(out, "End", "Header")

  hdr(out, "Begin", "Data " + format)


  hdr(out, "End", "Data " + format)
  hdr(out, "End", "Segment")
  
}

// Writes a header key/value pair to out:
// # Key: Value
func hdr(out io.Writer, key string, value interface{}){
  fmt.Fprintln(out, "# ", key, ": ", value)
}
