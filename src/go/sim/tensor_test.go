//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import (
	"testing"
	"tensor"
	"fmt"
)

func TestCopy(t *testing.T) {
	size := []int{4, 8, 16}
	dev1, dev2 := NewTensor(backend, size), NewTensor(backend, size)
	host1, host2 := tensor.NewTensorN(size), tensor.NewTensorN(size)

	for i := range host1.List() {
		host1.List()[i] = float(i)
	}

	TensorCopyTo(host1, dev1)
	TensorCopyOn(dev1, dev2)
	TensorCopyFrom(dev2, host2)

	assert(tensor.Equals(host1, host2))
}

func TestGetSet(t *testing.T) {
	tens := NewTensor(backend, []int{5, 6, 7})
	fmt.Println(tens)
	tens.Set([]int{4, 5, 6}, 3.14)
	if tens.Get([]int{4, 5, 6}) != 3.14 {
		t.Fail()
	}
}

func TestComponent(t *testing.T) {
	m := NewTensor(backend, []int{3, 10, 20})

	mx := m.Component(0)
	my := m.Component(1)
	mz := m.Component(2)

	my.Set([]int{0, 0}, 1.)
	my.Set([]int{9, 19}, 1.)

	if mx.Get([]int{9, 19}) != 0 {
		t.Fail()
	}
	if my.Get([]int{0, 0}) != 1 {
		t.Fail()
	}
	if my.Get([]int{9, 19}) != 1 {
		t.Fail()
	}
	if mz.Get([]int{0, 0}) != 0 {
		t.Fail()
	}
}
