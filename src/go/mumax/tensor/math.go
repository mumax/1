//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package tensor

// This file implements basic mathematical functions on tensors.

import ()

// Tests for deep equality (rank, size and data)
func Equal(a, b Interface) bool {

	sizeA, sizeB := a.Size(), b.Size()
	// test for equal rank
	if len(sizeA) != len(sizeB) {
		return false
	}

	// test for equal size
	for i, sa := range sizeA {
		if sa != sizeB[i] {
			return false
		}
	}

	// test for equal data
	dataA, dataB := a.List(), b.List()
	for i, da := range dataA {
		if da != dataB[i] {
			return false
		}
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

// Average of all elements
func Average(t Interface) float32 {
	l := t.List()
	sum := float64(0)
	for _, val := range l {
		sum += float64(val)
	}
	return float32(sum / float64(len(l)))
}


// Returns a component.
// I.e.: fixes the first index.
// Turns a A x B x C x ... tensor
// into a B x C x ... tensor.
// The underlying data is shared.
func Component(t Interface, comp int) *T {
	c := new(T)
	c.TSize = t.Size()[1:]
	length := Prod(c.TSize)
	start := comp * length
	stop := (comp + 1) * length
	c.TList = t.List()[start:stop]
	return c
}

// Returns a component without changing the rank.
// I.e.: fixes the first index and makes the first size 1.
// Turns a A x B x C x ... tensor
// into a 1 x B x C x ... tensor.
// The underlying data is shared.
func Component1(t Interface, comp int) *T {
	c := new(T)
	c.TSize = make([]int, Rank(t))
	copy (c.TSize, t.Size())
	c.TSize[0] = 1
	length := Prod(c.TSize)
	start := comp * length
	stop := (comp + 1) * length
	c.TList = t.List()[start:stop]
	return c
}


