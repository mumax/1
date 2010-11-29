//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.


// outtools is a program for analyzing simulation output data.
// 
// usage: outtools --command arg1 arg2 --command arg1 ...
//
// The commands are processed in the order specified.
// They call the corresponding method (using reflection) and
// print the return value(s) as text.
// 
// One important command is --recover "errorvalue":
// If the command can not be executed for some reason
// (e.g. invalid or nonexisting input file), outtools
// will not crash but print "errovalue".
//
package main

import (
	"refsh"
	. "tensor"
	"fmt"
	. "math"
	"os"
	"iotool"
	"strings"
	"strconv"
	"sim"
)

// When a panic occurs, do not crash
// but output "value" instead.
func (m *Main) Recover(value string) {
	m.recover_val = value
}

func (m *Main) Print(fname string) {
	t := ReadF(fname)
	WriteAscii(os.Stdout, t)
}


func (m *Main) DrawSlice(fname string, slicepos int) {
	tensor := ToT4(ReadF(fname))

	size := make([]int, 4)
	for i := range size {
		size[i] = tensor.Size()[i]
	}
	size[1] = 1

	slice := NewT4(size)
	for c := 0; c < 3; c++ {
		for j := 0; j < size[2]; j++ {
			for k := 0; k < size[3]; k++ {
				slice.TArray[c][0][j][k] = tensor.TArray[c][slicepos][j][k]
			}
		}
	}

	outname := sim.RemoveExtension(fname) + ".png"
	out, err := os.Open(outname, os.O_WRONLY|os.O_CREATE, 0666)
	if err != nil {
		panic(err)
	}
	sim.PNG(out, slice)
}

// returns the number of times torque_z has gone over the threshold.
// once over the threshold, it has to be at least 100ps before a new switch is reported.
// firsttime = time of the first switch
func (m *Main) SwitchDetect(fname string, threshold float) (nswitch int, firsttime float, ok string) {
  in, err := os.Open(fname, os.O_RDONLY, 0666)
  if err != nil {
    panic(err)
  }

  // discard header
  header, _ := iotool.ReadLine(in)
  for strings.HasPrefix(header, "#") {
    header, _ = iotool.ReadLine(in)
  }

  line, eof := iotool.ReadLine(in)
  sincelast := 0.
  for !eof {
    words := strings.Fields(line)
//     torquex, _ := strconv.Atof(words[7])
//     torquey, _ := strconv.Atof(words[8])
    torquez, _ := strconv.Atof(words[9])
    time, _ := strconv.Atof(words[0])

    
    
    if (time-sincelast) > 100e-12 && torquez > threshold{
      nswitch++
      if firsttime == 0.{
        firsttime = time
      }
      sincelast = time
    }

    line, eof = iotool.ReadLine(in)
  }

  if nswitch > 0{
    ok = "#yes"
  }else{
    ok = "#no"
  }
  return
}


// Calculates the RMS response of the in-plane magnetization.
// Used for, e.g., magnetic resonance.
func (m *Main) InplaneRMS(fname string) (rms float64, ok string) {
	in, err := os.Open(fname, os.O_RDONLY, 0666)
	if err != nil {
		panic(err)
	}

	// discard header
	header, _ := iotool.ReadLine(in)
	for strings.HasPrefix(header, "#") {
		header, _ = iotool.ReadLine(in)
	}

	N := 0
	// 	rms := float64(0.)

	line, eof := iotool.ReadLine(in)
	for !eof {
		words := strings.Fields(line)
		mx, e1 := strconv.Atof(words[2])
		my, e2 := strconv.Atof(words[3])
		if e1 != nil {
			panic(e1)
		}
		if e2 != nil {
			panic(e2)
		}
		//     fmt.Println(mx, " ", my)
		rms += float64(mx*mx + my*my)
		N++

		line, eof = iotool.ReadLine(in)
	}

	rms /= float64(N)
	return Sqrt(rms), "# ok"
}


func (m *Main) CorePol(fname string) (maxMz float64, updown string) {
	data := ToT4(ReadF(fname))
	array := data.Array()
	mz := array[0]
	answer := float64(mz[0][0][0])
	for i := range mz {
		for j := range mz[i] {
			for k := range mz[i][j] {
				if Fabs(answer) < Fabs(float64(mz[i][j][k])) {
					answer = (float64(mz[i][j][k]))
				}
			}
		}
	}
	maxMz = answer
	if maxMz > 0. {
		updown = "#up"
	} else {
		updown = "#down"
	}
	return
}


// LEGACY: works on old .txt format only
// (where the rank entry is missing and assumed 4)
// Returns the vortex core polarization.
// (value of max out-of-plane magnetization and a string "#up" or "down")
// func (m *Main) CorePol4(fname string) (maxMz float64, updown string) {
// 	data := ToT4(FReadAscii4(fname))
// 	array := data.Array()
// 	mz := array[0]
// 	answer := float64(mz[0][0][0])
// 	for i := range mz {
// 		for j := range mz[i] {
// 			for k := range mz[i][j] {
// 				if Fabs(answer) < Fabs(float64(mz[i][j][k])) {
// 					answer = (float64(mz[i][j][k]))
// 				}
// 			}
// 		}
// 	}
// 	maxMz = answer
// 	if maxMz > 0. {
// 		updown = "#up"
// 	} else {
// 		updown = "#down"
// 	}
// 	return
// }


func main() {
	sh := refsh.New()
	sh.CrashOnError = true
	main_ := NewMain()
	sh.AddAllMethods(main_)

	commands, args := refsh.ParseFlags()
	for i := range commands {
		func() {

			defer func() {
				if main_.recover_val != "" {
					err := recover()
					if err != nil {
						fmt.Fprintln(os.Stderr, err)
						fmt.Println(main_.recover_val)
					}
				}
			}()
			ret := sh.Call(commands[i], args[i])
			if ret == nil {
				ret = []interface{}{main_.recover_val}
			}
			for _, r := range ret {
				fmt.Print(r, " ")
			}
			if len(ret) != 0 {
				fmt.Println()
			}
		}()

	}
}

type Main struct {
	recover_val string
}

func NewMain() *Main {
	return new(Main)
}
