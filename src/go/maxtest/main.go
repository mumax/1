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
	"mumax/tensor"
	"io/ioutil"
	. "strings"
	"os"
	"exec"
	"fmt"
	"mumax/omf"
	"math"
	"path"
	"flag"
)

var maxerror *float64 = flag.Float64("maxerror", 0, "Maximum difference with refernce solution.")

func main() {
	flag.Parse()
	args := flag.Args()
	if len(args) == 0 { // no arguments: scan current directory
		args = []string{"."}
	}

	status := NewStatus()
	for _, dir := range args {
		fileinfo, err := ioutil.ReadDir(dir)
		if err != nil {
			panic(err)
		}

		for _, info := range fileinfo {
			out := info.Name
			outdir := dir + "/" + out
			if info.IsDirectory() && HasSuffix(out, ".out") {
				ref := out[:len(out)-len(".out")] + ".ref"
				refdir := dir + "/" + ref
				if contains(fileinfo, ref) {
					status.Combine(compareDir(outdir, refdir))
				} else {
					// TODO: if the .ref exists but not the .out, then something is wrong
					// perhaps it should be reported.
					copydir(outdir, refdir)
				}
			}
		}
	}
	fmt.Println("\nTOTAL: ", status)
	if !status.Ok() {
		os.Exit(-1)
	}
}


func compareDir(out, ref string) (status *Status) {

	// Should any error occur then the testing program must not crash.
	// We consider this an failure of the unit test instead.
	defer func() {
		err := recover()
		if err != nil {
			status = NewStatus()
			status.FatalError = true
		}
	}()

	fmt.Print(out, ":\t")

	fileinfo, err := ioutil.ReadDir(ref)
	if err != nil {
		panic(err)
	}
	status = NewStatus()
	for _, info := range fileinfo {
		reffile := ref + "/" + info.Name
		outfile := out + "/" + info.Name
		status.Combine(compareFile(outfile, reffile))
	}

	fmt.Println(status)
	return
}


func compareFile(out, ref string) (status *Status) {

	// Should any error occur then the testing program must not crash.
	// We consider this an failure of the unit test instead.
	defer func() {
		err := recover()
		if err != nil {
			status = NewStatus()
			status.FatalError = true
		}
	}()

	// Compare the contents of files depending on their extension.
	switch {
	default:
		status = skip(out, ref)
	case HasSuffix(out, ".omf"):
		status = compareOmf(out, ref)
	case path.Base(out) == "running": // if file "running" is present, the simulation has crashed.
		status = NewStatus()
		status.FatalError = true
	}
	return
}

// Skip checking the contents becuase the file type is unknown.
// However, the file should at least exist in the directory being tested.
func skip(out, ref string) (status *Status) {
	status = NewStatus()
	status.Filecount = 1
	_, err := os.Stat(out)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		status.FatalError = true
	}
	return
}

// compare omf files and report the rms error
func compareOmf(out, ref string) *Status {
	_, dataA := omf.FRead(out)
	_, dataB := omf.FRead(ref)
	s := NewStatus()
	s.Filecount = 1
	if !tensor.EqualSize(dataA.Size(), dataB.Size()) {
		s.FatalError = true
		return s
	}
	a := dataA.List()
	b := dataB.List()
	for i := range a {
		s.MaxError += sqr(a[i] - b[i])
		//		err := abs32(a[i] - b[i])
		//		if err > s.MaxError {
		//			s.MaxError = err
		//		}
	}
	s.MaxError = sqrt(s.MaxError / float32(len(a)))
	return s
}

func abs32(a float32) float32 {
	if a > 0 {
		return a
	}
	return -a
}

func max32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}


func sqrt(a float32) float32 {
	return float32(math.Sqrt(float64(a)))
}

func sqr(a float32) float32 {
	return a * a
}

type Status struct {
	Filecount  int
	MaxError   float32
	FatalError bool
}

func (s *Status) Combine(s2 *Status) {
	s.Filecount += s2.Filecount
	s.MaxError = max32(s.MaxError, s2.MaxError)
	s.FatalError = s.FatalError || s2.FatalError
}

func (s *Status) String() string {
	ok := "[ ok ]"
	if !s.Ok() {
		ok = "[FAIL]"
	}
	return fmt.Sprintf("Files:%d\t error:%f \t%s", s.Filecount, s.MaxError, ok)
}


func NewStatus() *Status {
	s := new(Status)
	return s
}

func (s *Status) Ok() bool {
	return !s.FatalError && s.MaxError <= float32(*maxerror)
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
