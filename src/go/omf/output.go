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

const ()

// Encodes the vector field in omf format.
// The swap from ZYX (internal) to XYZ (external) is made here.
func (c *OmfCodec) Encode(out_ io.Writer, f Interface) {
	out := bufio.NewWriter(out_)
	defer out.Flush()



	tens, multiplier, valueunit := f.GetData()
	data := (tensor.ToT4(tens)).Array()
	vecsize := tens.Size()
  if len(vecsize) != 4 {
    panic("rank should be 4")
  }
  if vecsize[0] != 3 {
    panic("size[0] should be 3")
  }
  gridsize := vecsize[1:]
	cellsize, meshunit := f.GetMesh()

	format := "text"

	hdr(out, "OOMMF", "rectangular mesh v1.0")
	hdr(out, "Segment count", "1")
	hdr(out, "Begin", "Segment")

	hdr(out, "Begin", "Header")

	hdr(out, "Title", out_)
	hdr(out, "meshtype", "rectangular")

	hdr(out, "meshunit", meshunit)

	hdr(out, "xBase", 0)
	hdr(out, "yBase", 0)
	hdr(out, "zBase", 0)
  hdr(out, "xStepSize", cellsize[Z])
  hdr(out, "xStepSize", cellsize[Y])
  hdr(out, "xStepSize", cellsize[X])
  hdr(out, "xNodes", gridsize[Z])
  hdr(out, "yNodes", gridsize[Y])
  hdr(out, "zNodes", gridsize[X])

  
	hdr(out, "valueunit", valueunit)
	hdr(out, "valuemultiplier", multiplier)

	hdr(out, "End", "Header")

	hdr(out, "Begin", "Data "+format)

  for i:=0; i<gridsize[Z]; i++{
    for j:=0; j<gridsize[Y]; j++{
      for k:=0; k<gridsize[X]; k++{
        for c:=0; c<3; c++{
          fmt.Fprint(out, data[c][k][j][i], " ")
        }
        fmt.Fprint(out, "\t")
      }
      fmt.Fprint(out, "\n")
    }
    fmt.Fprint(out, "\n")
  }

	hdr(out, "End", "Data "+format)
	hdr(out, "End", "Segment")

}

// Writes a header key/value pair to out:
// # Key: Value
func hdr(out io.Writer, key string, value interface{}) {
	fmt.Fprintln(out, "# ", key, ": ", value)
}
