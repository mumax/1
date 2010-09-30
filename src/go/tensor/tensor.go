//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package tensor

import (
	"reflect"
)

// const (
// 	X = 0
// 	Y = 1
// 	Z = 2
// )

/** The tensor interface: get size and data */

type Tensor interface {
	Size() []int
	Get(index []int) float
}


/** Tensor rank = length of size array */

func Rank(t Tensor) int { return len(t.Size()) }

/// @deprecated use Len
func N(t Tensor) int {
	n := 1
	size := t.Size()
	for i := range size {
		n *= size[i]
	}
	return n
}

// func Len(t Tensor) int{
//     n := 1
//     size := t.Size()
//     for i := range (size) {
//         n *= size[i]
//     }
//     return n
// }

/** Variadic get, utility method. */

// func Get(t Tensor, index_vararg ... int) float {
// 	indexarr := ToIntArray(index_vararg)
// 	return t.Get(indexarr)
// }


/** Converts vararg to int array. */

func ToIntArray(varargs interface{}) []int {
	sizestruct := reflect.NewValue(varargs).(*reflect.StructValue)
	rank := sizestruct.NumField()
	size := make([]int, rank)
	for i := 0; i < rank; i++ {
		size[i] = sizestruct.Field(i).Interface().(int)
	}
	return size
}

/** Tests Tensor Equality */

func Equals(dest, source Tensor) bool {
	if !EqualSize(dest.Size(), source.Size()) {
		return false
	}
	for i := NewIterator(dest); i.HasNext(); i.Next() {
		if dest.Get(i.Index()) != source.Get(i.Index()) {
			return false
		}
	}
	return true
}


/** Tests if both int slices are equal, in which case they represent equal Tensor sizes. */

func EqualSize(a, b []int) bool {
	if len(a) != len(b) {
		return false
	} else {
		for i := range a {
			if a[i] != b[i] {
				return false
			}
		}
	}
	return true
}
