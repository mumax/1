//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import (
	"testing"
)

var Ns []int = []int{2, 8, 16, 31, 33, 128, 255, 511, 1023, 1024, 1025, 20000}

func TestSum(t *testing.T) {

	for _, N := range Ns {

		host := make([]float, N)
		for i := range host {
			host[i] = 1.
		}

		dev := backend.newArray(N)
		backend.memcpyTo(&(host[0]), dev, N)

		sum := NewSum(backend, N)
		result := sum.reduce_(dev)

		if result != float(N) {
			t.Error("expected ", N, " got ", result)
		}
	}
}

func TestMax(t *testing.T) {

	for _, N := range Ns {

		host := make([]float, N)
		for i := range host {
			host[i] = 1.
		}

		host[12349%N] = 10. //insert 10. in some quasi-random position

		dev := backend.newArray(N)
		backend.memcpyTo(&(host[0]), dev, N)

		max := NewMax(backend, N)
		result := max.reduce_(dev)

		if result != 10. {
			t.Error("expected ", 10., " got ", result)
		}
	}
}

func TestMaxAbs(t *testing.T) {

	for _, N := range Ns {

		host := make([]float, N)
		for i := range host {
			host[i] = 1.
		}

		host[12349%N] = -10. //insert 10. in some quasi-random position

		dev := backend.newArray(N)
		backend.memcpyTo(&(host[0]), dev, N)

		max := NewMaxAbs(backend, N)
		result := max.reduce_(dev)

		if result != 10. {
			t.Error("expected ", 10., " got ", result)
		}
	}
}
