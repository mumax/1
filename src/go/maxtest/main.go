//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

// Maxtest is a tool for automated testing of mumax simulation output.
// It is mainly intended to run large batches of simulations and detect regressions.
//
// A directory is scanned for .out subdirectories and corresponding .ref
// directories with reference output. If the .ref does not exist, then it
// is assumed that the test is run for the first time: the .out is copied
// to the .ref which will from now serve as reference data.
// All files in the .ref and .out with known formats (.omf,...) are read
// and compared. They should have equal contents within a small error margin.
// 
package main

import (
	"io/ioutil"
	. "strings"
	"os"
	"exec"
	"fmt"
)


func main() {
	dir := "."
	fileinfo, err := ioutil.ReadDir(dir)
	if err != nil {
		panic(err)
	}

	for _, info := range fileinfo {
		out := info.Name
		outdir := out + info.Name
		if info.IsDirectory() && HasSuffix(out, ".out") {
			ref := out[:len(out)-len(".out")] + ".ref"
			refdir := dir + "/" + ref
			if contains(fileinfo, ref) {
				compareDir(outdir, refdir)
			} else {
					// TODO: if the .ref exists but not the .out, then something is wrong
					// perhaps it should be reported.
				copydir(outdir, refdir)
			}
		}
	}
}


func compareDir(out, ref string) {
	fmt.Println(out, ":")

	fileinfo, err := ioutil.ReadDir(ref)
	if err != nil {
		panic(err)
	}
	for _, info := range fileinfo {
		reffile := ref + "/" + info.Name
		outfile := out + "/" + info.Name
		compareFile(outfile, reffile)
	}
}


func compareFile(out, ref string) *Status {
	switch {
	default:
		return skip(out, ref)
	case HasSuffix(out, ".omf"):
		return compareOmf(out, ref)
	}
	panic("Bug")
	return nil
}


func skip(out, ref string) *Status {
	return NewStatus() //empty: file skiped
}

func compareOmf(out, ref string) *Status {
	return NewStatus() //TODO
}

type Status struct {
	Filecount int
	MaxError  float32
}

func NewStatus() *Status {
	s := new(Status)
	return s
}


func copydir(src, dest string) {
	fmt.Println("cp -r ", src, " ", dest)
	args := []string{"cp", "-r", src, dest}

	wd, errwd := os.Getwd()
	if errwd != nil {
		panic(errwd)
	}

	cmd, err := exec.Run("/bin/cp", args, os.Environ(), wd, exec.PassThrough, exec.PassThrough, exec.MergeWithStdout)
	if err != nil {
		panic(err)
	}
	_, errw := cmd.Wait(0)
	if errw != nil {
		panic(errw)
	}
}


// Checks if the fileinfo array contains the named file
func contains(fileinfo []*os.FileInfo, file string) bool {
	for _, info := range fileinfo {
		if info.Name == file {
			return true
		}
	}
	return false
}
