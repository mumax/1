//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

// odttool is a general-purpose manipulator for .odt files.
// 
// General usage:
// odttool --command="arg1,arg2" ... infiles outfile
//
//
package main


import (
	. "mumax/common"
	"mumax/omf"
	"refsh"
	"fmt"
	"path"
	"os"
)


// Stores the currently loaded odt file.
var (
	filename string
	table    *omf.Table // the currently opened file
)


// Stores the table being built
var (
	newtable omf.Table
)


func GetDesc(key string) {
	desc := table.Desc[key]
	newtable.EnsureColumn(key, "")
	value := Atof32(fmt.Sprint(desc))
	newtable.AppendToColumn(key, value)
}

// CLI args consist of flags (starting with --) and files.
// They are passed like this:
// --command="arg1, arg2" ... file1 file2 ...
// The command is executed on each of the files
func main() {
	sh := refsh.New()
	sh.AddFunc("getdesc", GetDesc)
	cmd, args, files := refsh.ParseFlags2()

	// Each file is read and stored in "data".
	// Then, all commands are executed on that data.

	if len(files) == 0 {
		fmt.Fprintln(os.Stderr, "No input files")
		os.Exit(-1)
	}

	for _, file := range files {
		table = omf.ReadTable(MustOpenRDONLY(file))
		filename = file

		if len(cmd) == 0 {
			fmt.Fprintln(os.Stderr, "No commands")
			os.Exit(-1)
		}

		for i := range cmd {
			sh.Call(cmd[i], args[i])
		}
	}
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

