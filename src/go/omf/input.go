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
)

func (c *OmfCodec) Decode(in_ io.Reader) (t *tensor.T, metadata map[string]string){
  return Decode(in_)
}

func Decode(in_ io.Reader) (t *tensor.T, metadata map[string]string){
    in := bufio.NewReader(in_)
    metadata = ReadHeader(in)
    for k,v := range metadata{
      fmt.Println(k, ":", v)
    }
    return
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
    return HasPrefix(ToLower(Trim(str, "# ")), "end:data")
}

func ReadHeader(in_ io.Reader) map[string]string {
    header := make(map[string]string)
    in := bufio.NewReader(in_)
    line, eof := iotool.ReadLine(in)
    for !eof && !isHeaderEnd(line) {
        //if line == "" {continue}
        key, value := parseHeaderLine(line)
        header[key] = value
        line, eof = iotool.ReadLine(in)
    }
    return header
}

