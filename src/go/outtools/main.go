//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package main

import (
	"refsh"
	. "tensor2"
	"fmt"
	. "math"
	"os"
)


func main() {
	sh := refsh.New()
	main_ := NewMain()
	sh.AddAllMethods(main_)

	commands, args := refsh.ParseFlags()
	for i := range commands {
    
		func() {

      defer func() {
				if main_.recover_val != "" {
          err := recover()
					if err != nil { fmt.Fprintln(os.Stderr, err); fmt.Println(main_.recover_val)}
				}
			}()

			ret := sh.Call(commands[i], args[i])
			if ret == nil {
				ret = []interface{}{main_.recover_val}
			}
			for _, r := range ret {
				fmt.Print(r, " ")
			}
			if len(ret) != 0 {fmt.Println()}
		}()
		
	}
}

func (m *Main) Recover(value string) {
	m.recover_val = value
}


// returns the vortex core polarization
// (value of max out-of-plane magnetization and a string "#up" or "down")
func (m *Main) CorePol(fname string) (maxMz float64, updown string) {
	data := ToT4(FReadAscii(fname))
	array := data.Array
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


type Main struct {
	recover_val string
}

func NewMain() *Main {
	return new(Main)
}
