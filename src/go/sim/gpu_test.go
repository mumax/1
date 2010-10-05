//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import (
	"testing"
	"tensor"
)

var backend = GPU //Backend{NewRemoteDevice("127.0.0.1", ":2527"), false}

func TestPad(t *testing.T) {
	small := []int{4, 8, 16}
	big := []int{6, 12, 32}

	dev1, dev2 := NewTensor(backend, small), NewTensor(backend, big)
	host1, host2 := tensor.NewTensorN(small), tensor.NewTensorN(small)

	for i := range host1.List() {
		host1.List()[i] = float(i)
	}

	TensorCopyTo(host1, dev1)
	CopyPad(dev1, dev2)
	ZeroTensor(dev1)
	CopyUnpad(dev2, dev1)
	TensorCopyFrom(dev1, host2)

	assert(tensor.Equals(host1, host2))
}


func TestStride(t *testing.T) {
	// 	s := backend.Stride()
	// 	backend.OverrideStride(10)
	// 	if backend.Stride() != 10 {
	// 		t.Fail()
	// 	}
	// 
	// 	for i := 1; i < 100; i++ {
	// 		if backend.PadToStride(i)%backend.Stride() != 0 {
	// 			t.Fail()
	// 		}
	// 	}
	// 
	// 	backend.OverrideStride(-1)
	// 	if backend.Stride() != s {
	// 		t.Fail()
	// 	}
}


// func TestMisc(t *testing.T) {
// 	backend.PrintProperties()
// }


func TestZero(t *testing.T) {
	N := 100
	host := make([]float, N)
	dev := backend.newArray(N)

	for i := range host {
		host[i] = float(i)
	}

	backend.memcpyTo(&host[0], dev, N)
	backend.zero(dev, N/2)
	backend.memcpyFrom(dev, &host[0], N)

	for i := 0; i < N/2; i++ {
		if host[i] != 0. {
			t.Fail()
		}
	}
	for i := N / 2; i < N; i++ {
		if host[i] != float(i) {
			t.Fail()
		}
	}
}


func TestMemory(t *testing.T) {
	N := 100
	host1, host2 := make([]float, N), make([]float, N)
	dev1, dev2 := backend.newArray(N), backend.newArray(N)

	for i := range host1 {
		host1[i] = float(i)
	}

	backend.memcpyTo(&host1[0], dev1, N)
	backend.memcpyOn(dev1, dev2, N)
	backend.memcpyFrom(dev2, &host2[0], N)

	for i := range host1 {
		if host1[i] != host2[i] {
			t.Fail()
		}
	}
}
