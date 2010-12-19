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

/*
Example: 
# OOMMF: rectangular mesh v1.0
# Segment count: 1
# Begin: Segment
# Begin: Header
# Title: oxs-demo-8920.omf
# Desc: This is debug output from the prototype oxs solver
# meshtype: rectangular
# meshunit: m
# xbase: 1.5e-09
# ybase: 1.5e-09
# zbase: 1.5e-09
# xstepsize: 3e-09
# ystepsize: 3e-09
# zstepsize: 3e-09
# xnodes: 15
# ynodes: 15
# znodes: 15
# xmin: 0
# ymin: 0
# zmin: 0
# xmax: 4.5e-08
# ymax: 4.5e-08
# zmax: 4.5e-08
# boundary: 0 0 2.25e-08 0 4.5e-08 2.25e-08 4.5e-08 4.5e-08 2.25e-08 4.5e-08 0 2.25e-08 0 0 2.25e-08
# valueunit: A/m
# valuemultiplier: 1
# ValueRangeMinMag: 9.9899999999999988e-09
# ValueRangeMaxMag: 1
# End: Header
# Begin: Data Binary 4
*/
func (c *OmfCodec) Encode(out_ io.Writer, t tensor.Interface, metadata map[string]string) {
  out := bufio.NewWriter(out_)

  hdr(out, "OOMMF", "rectangular mesh v1.0")
  hdr(out, "Segment count", "1")
  hdr(out, "Begin", "Segment")
  hdr(out, "Begin", "Header")
  hdr(out, "Title", fmt.Sprint(out_))
  hdr(out, "Desc", "")
  hdr(out, "meshtype", "rectangular")
  hdr(out, "meshunit", "m")
  
  
}

// Writes a header key/value pair to out:
// # Key: Value
func hdr(out io.Writer, key, value string){
  
}