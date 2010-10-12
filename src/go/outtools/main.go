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
)


func main() {
	sh := refsh.New()
	main_ := NewMain()
	sh.AddAllMethods(main_)

	commands, args := refsh.ParseFlags()
	for i := range commands {
		ret := sh.Call(commands[i], args[i])
		for _,r := range ret{
      fmt.Print(r, " ")
    }
    fmt.Println()
	}
}


func (m *Main) CorePol(fname string) float64 {
	data := ToT4(FReadAscii(fname))
	array := data.Array
	mz := array[0]
	answer := float64(mz[0][0][0])
	for i := range mz {
		for j := range mz[i] {
			for k := range mz[i][j] {
				if Fabs(answer) < Fabs(float64(mz[i][j][k])) {
					answer = Fabs(float64(mz[i][j][k]))
				}
			}
		}
	}
	return answer
}


type Main struct {
	
}

func NewMain() *Main {
	return new(Main)
}
