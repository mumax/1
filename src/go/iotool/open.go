//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package iotool

import (
    "io"
    "os"
)

func MustOpenRDONLY(filename string) os.File{
  file, err := os.Open(filename, os.O_RDONLY, 0777)
  if err != nil{
    panic(err)
  }
  return file
}

func MustOpenWRONLY(filename string) os.File{
  
}

func Parent(filename string) string{

}