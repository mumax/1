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
#
## This is a comment.
## No comments allowed in the first line.
#
# Segment count: 1
## Number of segments. Should be 1 for now.
#
# Begin: Segment
# Begin: Header
#
# Title: Long file name or title goes here
#
# Desc: ’Description’ tag, which may be used or ignored by postprocessing
# Desc: programs. You can put anything you want here, and can have as many
# Desc: ’Desc’ lines as you want. The ## comment marker is disabled in
# Desc: description lines.
#
## Fundamental mesh measurement unit. Treated as a label:
# meshunit: nm
#
# meshtype: rectangular
# xbase: 0.
## (xbase,ybase,zbase) is the position, in
# ybase: 0.
## ’meshunit’, of the first point in the data
# zbase: 0.
## section (below).
#
# xstepsize: 20. ## Distance between adjacent grid pts.: on the x-axis,
158
# ystepsize: 10. ## 20 nm, etc. The sign on this value determines the
# zstepsize: 10. ## grid orientation relative to (xbase,ybase,zbase).
#
# xnodes: 200
## Number of nodes along the x-axis, etc. (integers)
# ynodes: 400
# znodes: 1
#
# xmin: 0.
## Corner points defining mesh bounding box in
# ymin:
0.
## ’meshunit’. Floating point values.
# zmin: -10.
# xmax: 4000.
# ymax: 4000.
# zmax: 10.
#
## Fundamental field value unit, treated as a label:
# valueunit: kA/m
# valuemultiplier: 0.79577472 ## Multiply data block values by this
#
## to get true value in ’valueunits’.
#
# ValueRangeMaxMag: 1005.3096 ## These are in data block value units,
# ValueRangeMinMag: 1e-8
## and are used as hints (or defaults)
#
## by postprocessing programs. The mmDisp program ignores any
#
## points with magnitude smaller than ValueRangeMinMag, and uses
#
## ValueRangeMaxMag to scale inputs for display.
#
# End: Header
#
## Anything between ’# End: Header’ and ’# Begin: data text’,
## ’# Begin: data binary 4’ or ’# Begin: data binary 8’ is ignored.
##
## Data input is in ’x-component y-component z-component’ triples,
## ordered with x incremented first, then y, and finally z.
#
# Begin: data text
1000 0 0 724.1 0. 700.023
578.5 500.4 -652.36
<...data omitted for brevity...>
252.34 -696.42 -671.81
# End: data text
# End: segment

*/
func (c *OmfCodec) Encode(out_ io.Writer, t tensor.Interface, metadata map[string]string) {
  out := bufio.NewWriter(out_)
  format := "text"

  hdr(out, "OOMMF", "rectangular mesh v1.0")
  hdr(out, "Segment count", "1")
  hdr(out, "Begin", "Segment")
  
  hdr(out, "Begin", "Header")
  
  hdr(out, "Title", fmt.Sprint(out_))
  hdr(out, "meshtype", "rectangular")
  hdr(out, "meshunit", "m")

  hdr(out, "End", "Header")

  hdr(out, "Begin", "Data " + format)


  hdr(out, "End", "Data " + format)
  hdr(out, "End", "Segment")
  
}

// Writes a header key/value pair to out:
// # Key: Value
func hdr(out io.Writer, key, value string){
  
}