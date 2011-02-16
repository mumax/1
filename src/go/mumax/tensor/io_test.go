//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package tensor


import (
	"testing"
)

func TestIO(test *testing.T) {
	size := []int{3, 15, 19, 31}
	t1 := NewT4(size)
	for i := range t1.List(){
		t1.List()[i] = float32(i)
	}	
	WriteF("iotest.t", t1)
	t2 := ReadF("iotest.t")
	if ! Equal(t1, t2){test.Fail()}
}
