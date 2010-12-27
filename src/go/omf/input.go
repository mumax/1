//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package omf

import (
	"io"
	"tensor"
	"iotool"
    . "strings"
    "bufio"
    "fmt"
    "strconv"
)

func (c *OmfCodec) Decode(in_ io.Reader) (t *tensor.T, metadata map[string]string){
  return Decode(in_)
}

func Decode(in_ io.Reader) (t *tensor.T, metadata map[string]string){
    in := bufio.NewReader(in_)
    info := ReadHeader(in)
    metadata = info.Desc
    for k,v := range metadata{
      fmt.Println(k, ":", v)
    }
    return
}


// omf.Info represents the header part of an omf file.
type Info struct{
  Desc map[string]string
  Size [3]int
}


// INTERNAL: Splits "# key: value" into "key", "value"
func parseHeaderLine(str string) (key, value string) {
    strs := Split(str, ":", 2)
    key = Trim(strs[0], "# ")
    value = Trim(strs[1], "# ")
    return
}

// INTERNAL: true if line == "# begin_data"
func isHeaderEnd(str string) bool {
  str = ToLower(Trim(str, "# "))
  str = Replace(str, " ", "", -1)
//   fmt.Println(str)
    return HasPrefix(str, "begin:data")
}

func ReadHeader(in io.Reader) *Info {
    desc := make(map[string]string)
    info := new(Info)
    info.Desc = desc
 
    line, eof := iotool.ReadLine(in)
    for !eof && !isHeaderEnd(line) {
        key, value := parseHeaderLine(line)
        
        switch ToLower(key){
          default: panic("Unknown key: " + key)
              // ignored
          case "oommf", "segment count", "begin", "title", "meshtype", "xbase", "ybase", "zbase", "xstepsize", "ystepsize", "zstepsize", "xmin", "ymin", "zmin", "xmax", "ymax", "zmax", "valuerangeminmag", "valuerangemaxmag", "end":
          case "xnodes": info.Size[X] = atoi(value)
          case "ynodes": info.Size[Y] = atoi(value)
          case "znodes": info.Size[Z] = atoi(value)
          case "valuemultiplier":
          case "valueunit":
          case "meshunit":
            // desc tags: parse further and add to metadata table
            // TODO: does not neccesarily contain a ':'
          case "desc": 
            strs := Split(value, ":", 2)
            desc_key := Trim(strs[0], "# ")
            desc_value := Trim(strs[1], "# ")
            desc[desc_key] = desc_value

        }
        
        line, eof = iotool.ReadLine(in)
    }
    return info
}

func atoi(a string) int{
  i, err := strconv.Atoi(a)
  if err != nil{
    panic(err)
  }
  return i
}