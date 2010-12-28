//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

// omftools is a general-purpose manipulator for .omf files
//
package main


import (
	"refsh"
// 	"fmt"
	"tensor"
	"omf"
	"iotool"
	"path"
	"draw"
)

var (
	filename string     // the currently opened file
	data     *tensor.T  // the currently opened data
	info     *omf.Info
)


// CLI args consist of flagss (starting with --) and files.
// They are passed like this:
// --command="arg1, arg2" ... file1 file2 ...
// The command is executed on each of the files
func main() {
	sh := refsh.New()
	sh.AddFunc("draw", Draw)
	cmd, args, files := refsh.ParseFlags2()
	// Each file is read and stored in "data".
	// Then, all commands are executed on that data.
	for _, file := range files {
		t4, _ := omf.Decode(iotool.MustOpenRDONLY(file))
		filename = file
		data = tensor.ToT(t4)

		for i := range cmd {
			sh.Call(cmd[i], args[i])
		}
	}
}


func Draw() {
  outfile := replaceExt(filename, ".png")
  out := iotool.MustOpenWRONLY(outfile)
  defer out.Close()
  draw.PNG(out, data)
}

func replaceExt(filename, newext string) string{
  extension := path.Ext(filename)
  return filename[:len(filename)-len(extension)] + newext
}

// func Slice(dirstr string, pos int){
//   dirstr = strings.ToUpper(dirstr)
//   var dir int
//   switch dirstr{
//     default: panic("Slice direction should be X, Y or Z")
//     case "X": dir = 0
//     case "Y": dir = 1
//     case "Z": dir = 2
//   }
// 
//   size := copy(data.Size())
// }
