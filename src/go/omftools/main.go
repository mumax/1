//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

// omftool is a general-purpose manipulator for .omf files.
// 
// General usage:
// omftool --command="arg1,arg2" ... files
//
// omftool loads the specified files in a buffer (one at a time),
// and calls all the commands on that buffer.
// The (possibley modified) buffer may be saved again.
//
package main


import (
	"refsh"
	"fmt"
	"tensor"
	"omf"
	"iotool"
	"path"
	"os"
)


// Stores the currently loaded omf file.
var (
	filename string     // the currently opened file
	data     *tensor.T4 // the currently opened vector data
	info     *omf.Info  // 
)


// CLI args consist of flagss (starting with --) and files.
// They are passed like this:
// --command="arg1, arg2" ... file1 file2 ...
// The command is executed on each of the files
func main() {
	sh := refsh.New()
	sh.AddFunc("draw", Draw)
	sh.AddFunc("draw3d", Draw3D)
    sh.AddFunc("draw3d-size", Draw3D_Size)
    sh.AddFunc("draw3d-detail", Draw3D_Detail)
	sh.AddFunc("downsample", Downsample)
	cmd, args, files := refsh.ParseFlags2()

	// Each file is read and stored in "data".
	// Then, all commands are executed on that data.

	if len(files) == 0 {
		fmt.Fprintln(os.Stderr, "No input files")
		os.Exit(-1)
	}

	for _, file := range files {
		t4, _ := omf.Decode(iotool.MustOpenRDONLY(file))
		filename = file
		data = t4 //tensor.ToT(t4)

		if len(cmd) == 0 {
			fmt.Fprintln(os.Stderr, "No commands")
			os.Exit(-1)
		}

		for i := range cmd {
			sh.Call(cmd[i], args[i])
		}
	}
}


func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// replaces the extension of filename by a new one.
func replaceExt(filename, newext string) string {
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

const (
	X = 0
	Y = 1
	Z = 2
)
