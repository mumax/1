//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package tensor


import (
	"testing"
)

func TestComponent(test *testing.T) {
	size4 := []int{3, 4, 5, 6}
	t := NewT4(size4)
	for i := range t.TArray {
		t.TArray[i][0][0][0] = float32(i)
	}
	c := Component(t, 1)
	if Rank(c) != 3 {
		test.Fail()
	}
	if c.TList[0] != 1 {
		test.Fail()
	}
	c1 := Component1(t, 1)
	if Rank(c1) != 4 {
		test.Fail()
	}
	if c1.TList[0] != 1 {
		test.Fail()
	}
}
