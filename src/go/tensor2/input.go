//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package tensor2

import (
	"io"
	"os"
	"bufio"
	"encoding/binary"
	"iotool"
	"strings"
)


// INTERNAL
// Reads a tensor header.
// Returns a map with the key/value pairs in the header
func ReadHeader(in_ io.Reader) map[string]string {
  header := make(map[string]string)
	in := bufio.NewReader(in_)
	line, eof := iotool.ReadLine(in)
	for !eof && !isHeaderEnd(line){
		key, value := parseHeaderLine(line)
 		header[key] = value
		line, eof = iotool.ReadLine(in)
	}
	return header
}

// INTERNAL: Splits "# key: value" into "key", "value"
func parseHeaderLine(str string) (key, value string) {
	strs := strings.Split(str, H_SEPARATOR, 2)
	key = strings.Trim(strs[0], "# ")
	value = strings.Trim(strs[1], "# ")
	return
}

// INTERNAL: true if line == "# begin_data"
func isHeaderEnd(str string) bool{
  return strings.Trim(str, "# ") == H_END
}


// INTERNAL
func ReadDataBinary(in_ io.Reader, t Interface) (err os.Error) {
	list := t.List()
	in := bufio.NewReader(in_)
	err = binary.Read(in, ENDIANESS, list)
	return
}
