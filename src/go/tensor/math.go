//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package tensor

import ()

// Tests for deep equality (rank, size and data)
func Equal(a, b Interface) bool{
  
  sizeA, sizeB := a.Size(), b.Size()
  // test for equal rank
  if len(sizeA) != len(sizeB) {return false}
  
  // test for equal size
  for i,sa := range sizeA{
    if sa != sizeB[i]{ return false}
  }
  
  // test for equal data
  dataA, dataB := a.List(), b.List()
  for i, da := range dataA{
    if da != dataB[i] {return false}
  }
  
  return true
}


// Finds the extrema.
func MinMax(t Interface) (min, max float32) {
	l := t.List()
	min, max = l[0], l[0]
	for _, val := range l {
		if val < min {
			min = val
		}
		if val > max {
			max = val
		}
	}
	return
}
