//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package refsh

import (
	"io"
)

// Reads one character from the Reader.
// -1 means EOF.
// Errors are cought and cause panic
func ReadChar(in io.Reader) int {
	buffer := [1]byte{}
	switch nr, err := in.Read(buffer[0:]); true {
	case nr < 0: // error
		panic(err)
	case nr == 0: // eof
		return -1
	case nr > 0: // ok
		return int(buffer[0])
	}
	return 0 // never reached
}

//
func ReadLine(in io.Reader) (line string, eof bool) {
  char := ReadChar(in)
  eof = isEOF(char)
  
  for !isEndline(char){
    line += string(byte(char))
    char = ReadChar(in)
  }
  return line, eof
}


func isEOF(char int) bool {
	return char == -1
}

func isEndline(char int) bool {
	return isEOF(char) || char == int('\n')
}
