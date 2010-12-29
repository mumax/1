//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package main

// This file implements functions for resizing the vector field
// (sliceing, averaging, cropping, etc.)

import (
    "tensor"
)


// Creates a lower-resolution vector field, scaled down by a factor f
func Downsample(f int) {
    bigsize := data.Size()
    smallsize := []int{3, bigsize[1] / f, bigsize[2] / f, bigsize[3] / f}
    for i := range smallsize {
        if smallsize[i] < 1 {
            smallsize[i] = 1
        }
    }
    small := tensor.NewT4(smallsize)
    A := data.Array()  // big array
    a := small.Array() // small array
    for c := range a {

        for i := range a[c] {
            for j := range a[c][i] {
                for k := range a[c][i][j] {

                    n := 0

                    for I := i * f; I < min((i+1)*f, bigsize[1]); I++ {
                        for J := j * f; J < min((j+1)*f, bigsize[2]); J++ {
                            for K := k * f; K < min((k+1)*f, bigsize[3]); K++ {
                                n++
                                a[c][i][j][k] += A[c][I][J][K]

                            }
                        }
                    }
                    a[c][i][j][k] /= float32(n)
                }
            }
        }
    }

    data = small
    //  TODO info.Gridsize = f*original gridsize (unless 1)
}

