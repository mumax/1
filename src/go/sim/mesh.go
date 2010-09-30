//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

type Mesh struct {
	size       [3]int   // Mesh Size, e.g. 4x64x64 TODO get rid of: already in FFT
	size4D     [4]int   // Vector-field-size of, a.o., the magnetization. 3 x N0 x N1 x N2
	paddedsize []int    // Mesh size with zero padding.
	cellSize   [3]float // Cell Size in exchange lengths, e.g. Inf x 0.5 x 0.5
}
