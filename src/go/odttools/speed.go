//  This file is part of MuMax, a high-performance micromagnetic simulator
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package main

import (
	. "mumax/common"
	"os"
	"math"
	"fmt"
)

func Diff(data string, timecol string) {
	col := table.GetColumn(data)
	time := table.GetColumn(timecol)
	diffname := "d" + data + "_d" + timecol
	newtable.EnsureColumn(diffname, "("+table.GetUnit(data)+")/("+table.GetUnit(timecol)+")")

	var prevTime float32 = 0
	var prevData float32 = 0
	for i := range col {
		data := col[i]
		t := time[i]
		var speed float32 = 0
		if i > 0 {
			dt := t - prevTime
			dx := data - prevData
			speed = dx / dt
		}
		prevTime = t
		prevData = data
		newtable.AppendToColumn(diffname, speed)
	}
}

func Diff2(data1, data2 string, timecol string) {
	col1 := table.GetColumn(data1)
	col2 := table.GetColumn(data2)
	time := table.GetColumn(timecol)
	diffname := "d" + data1 + "_d" + timecol
	newtable.EnsureColumn(diffname, "("+table.GetUnit(data1)+")/("+table.GetUnit(timecol)+")")

	var prevTime float32
	var prevData1, prevData2 float32
	for i := range col1 {
		data1 := col1[i]
		data2 := col2[i]
		t := time[i]
		var speed float32 = 0
		if i > 0 {
			dt := t - prevTime
			dx := data1 - prevData1
			dy := data2 - prevData2
			speed = Sqrt32(dx*dx+dy*dy) / dt
		}
		prevTime = t
		prevData1 = data1
		prevData2 = data2
		newtable.AppendToColumn(diffname, speed)
	}
}

func AvgDiff2(data1, data2 string, timecol string, max float32) {
	defer func() {
		err := recover()
		if err != nil {
			fmt.Fprintln(os.Stdout, err)
		}
	}()

	col1 := table.GetColumn(data1)
	col2 := table.GetColumn(data2)
	time := table.GetColumn(timecol)
	diffname := "d" + data1 + "_d" + timecol
	newtable.EnsureColumn(diffname, "("+table.GetUnit(data1)+")/("+table.GetUnit(timecol)+")")

	var prevTime float32
	var prevData1, prevData2 float32
	var total float64 = 0.
	var N int
	for i := range col1 {
		data1 := col1[i]
		data2 := col2[i]
		t := time[i]
		var speed float32 = 0
		if i > 0 {
			dt := t - prevTime
			dx := data1 - prevData1
			dy := data2 - prevData2
			speed = Sqrt32(dx*dx+dy*dy) / dt
		}
		prevTime = t
		prevData1 = data1
		prevData2 = data2
		if speed < max { // leave out bad peaks
			total += float64(speed)
			N++
		}
	}
	newtable.AppendToColumn(diffname, float32(total/float64(N)))
}

func AvgDiff2NoPeak(data1, data2 string, timecol string, peakcolName string, max float32) {
	defer func() {
		err := recover()
		if err != nil {
			fmt.Fprintln(os.Stdout, err)
		}
	}()

	peakcol := table.GetColumn(peakcolName)
	col1 := table.GetColumn(data1)
	col2 := table.GetColumn(data2)
	time := table.GetColumn(timecol)
	diffname := "d" + data1 + "_d" + timecol
	newtable.EnsureColumn(diffname, "("+table.GetUnit(data1)+")/("+table.GetUnit(timecol)+")")

	var prevTime float32
	var prevData1, prevData2 float32
	var total float32 = 0.
	//var N int
	peak := false
	for i := range col1 {
		data1 := col1[i]
		data2 := col2[i]
		t := time[i]
		var speed float32 = 0
		if peakcol[i] > max {
			peak = true
			break
		}
		if i > 0 {
			dt := t - prevTime
			dx := data1 - prevData1
			dy := data2 - prevData2
			speed = Sqrt32(dx*dx+dy*dy) / dt
		}
		prevTime = t
		prevData1 = data1
		prevData2 = data2
		if speed < 500 { // leave out bad peaks hack
			if speed > total {
				total = speed
			}
			//N++
		}
	}
	speed := float32(total) //float64(N))
	if peak {
		speed = 0
	}
	newtable.AppendToColumn(diffname, speed)
}


func InplaneRMS(data1, data2 string, peakcolName string, max float32) {
	defer func() {
		err := recover()
		if err != nil {
			fmt.Fprintln(os.Stdout, err)
		}
	}()

	peakcol := table.GetColumn(peakcolName)
	col1 := table.GetColumn(data1)
	col2 := table.GetColumn(data2)
	newtable.EnsureColumn("inplanerms", "")
	var rms float64
	var N int
	peak := false
	for i := range col1 {
		data1 := col1[i]
		println(data1)
		data2 := col2[i]
		println(data2)
		rms += float64(data1*data1 + data2*data2)
		N++
		if peakcol[i] > max {
			peak = true
			break
		}
	}
	rms = math.Sqrt(rms / float64(N))
	if peak {
		rms = 0
	}
	newtable.AppendToColumn("inplanerms", float32(rms))
}
