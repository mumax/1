//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package main

// This file implements functions for drawing the magnetization state

import (
	. "mumax/common"
	"mumax/tensor"
	"mumax/draw"
	"os"
	"fmt"
	"exec"
	"io"
)

// Renders in 2D, automatically saves in a .png file.
func Draw() {
	outfile := replaceExt(filename, ".png")
	if !FileExists(outfile) {
		out := MustOpenWRONLY(outfile)
		defer out.Close()
		draw.PNG(out, data.Component(c))
	} else {
		fmt.Fprintln(os.Stderr, "File exists:", outfile)
	}
}


// Renders in 2D, automatically saves in a .png file.
func DrawComp(c int) {
	compname = string('x' + c)
	outfile := replaceExt(filename, "_" + compname + ".png")
	if !FileExists(outfile) {
		out := MustOpenWRONLY(outfile)
		defer out.Close()
		draw.PNG(out, data)
	} else {
		fmt.Fprintln(os.Stderr, "File exists:", outfile)
	}
}

// Parameter for draw3d(), passed on to maxview
var (
	draw3d_zoom   int     = 64
	draw3d_detail int     = 32
	draw3d_shadow float32 = 0.8
)


func Draw3D_Size(arrowsize int) {
	draw3d_zoom = arrowsize
}

func Draw3D_Detail(vertices int) {
	draw3d_detail = vertices
}

func Draw3D_Shadow(shadow float32) {
	draw3d_shadow = shadow
}

// Renders in 3D, automatically savesin a .png file.
// This function depends on the java program "maxview".
func Draw3D() {
	outfile := replaceExt(filename, ".png")

	// Fork a maxview instance 
	wd, err1 := os.Getwd()
	if err1 != nil {
		panic(err1)
	}
	executable, err0 := exec.LookPath("maxview")
	if err0 != nil {
		panic(err0)
	}
	cmd, err := exec.Run(executable, []string{}, os.Environ(), wd, exec.Pipe, exec.PassThrough, exec.PassThrough)
	if err != nil {
		panic("running maxview: " + err.String())
	}

	// Pipe commands to maxview's stdin
	draw3D_Commands(cmd.Stdin)
	fmt.Fprintf(cmd.Stdin, "save %s\n", outfile)
	fmt.Fprintf(cmd.Stdin, "exit\n")

	// Wait for maxview to finish rendering
	_, err3 := cmd.Wait(0)
	if err3 != nil {
		panic(err3)
	}
}

// Interactive
func Draw3D_Dump() {
	draw3D_Commands(os.Stdout)
	fmt.Fprintf(os.Stdout, "show\n")
}

// INTERNAL
func draw3D_Commands(stdin io.Writer) {

	// Pipe commands to maxview's stdin
	zoom := draw3d_zoom // pixels per cone
	fmt.Fprintf(stdin, "size %d %d \n", zoom*data.Size()[3], zoom*data.Size()[2])
	fmt.Fprintf(stdin, "detail %d\n", draw3d_detail)
	fmt.Fprintf(stdin, "shadow %f\n", draw3d_shadow)

	a := tensor.ToT4(data).Array()
	imax := len(a[X])
	jmax := len(a[X][0])
	kmax := len(a[X][0][0])
	for i := 0; i < imax; i++ {
		for j := 0; j < jmax; j++ {
			for k := 0; k < kmax; k++ {
				x := float32(k) - float32(kmax)/2 + .5
				y := float32(j) - float32(jmax)/2 + .5
				z := float32(i) - float32(imax)/2 + .5
				fmt.Fprintf(stdin, "vec %f %f %f %f %f %f\n", x, y, z, a[Z][i][j][k], a[Y][i][j][k], a[X][i][j][k])
			}
		}
	}

}
