//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package tensor2

import (
	"io"
	"os"
	"fmt"
	"bufio"
)


func Prod(size []int) int {
	prod := 1
	for _, s := range size {
		prod *= s
	}
	return prod
}

type T struct {
	Size []int
	List []float
}

func NewT(size []int) *T {
	t := new(T)
	t.Init(size)
	return t
}

func (t *T) Init(size []int) {
	t.Size = size
	t.List = make([]float, Prod(size))
}

// TEMP HACK: RANK IS NOT STORED IN ASCII FORMAT
// ASSUME 4
func ReadAscii(in_ io.Reader) *T {
	rank := 4
// 	_, err := fmt.Fscan(in, &rank)
// 	if err != nil {
// 		panic(err)
// 	}

  in := bufio.NewReader(in_)

	size := make([]int, rank)
	for i := range size {
		_, err := fmt.Fscan(in, &size[i])
		if err != nil {
			panic(err)
		}
	}

	t := NewT(size)
	list := t.List

	for i := range list {
		_, err := fmt.Fscan(in, &list[i])
		if err != nil {
			panic(err)
		}
	}

	return t
}

func FReadAscii(fname string) *T {
	in, err := os.Open(fname, os.O_RDONLY, 0666)
	defer in.Close()
	if err != nil {
		panic(err)
	}
	return ReadAscii(in)
}


type T4 struct {
	T
	Array [][][][]float
}

func (t *T4) Init(size []int) {
	if len(size) != 4 {
		panic("Illegal argument")
	}
	t.Size = size
	t.List, t.Array = Array4D(size[0], size[1], size[2], size[3])
}

func NewT4(size []int) *T4 {
	t := new(T4)
	t.Init(size)
	return t
}


func ToT4(t *T) *T4 {
	return &T4{*t, Slice4D(t.List, t.Size)}
}


// Allocates a 2D array, as well as the contiguous 1D array backing it.
func Array2D(size0, size1 int) ([]float, [][]float) {
	if !(size0 > 0 && size1 > 0) {
		panic("Array size must be > 0")
	}

	// First make the slice and then the list. When the memory is not fragmented,
	// they are probably allocated in a good order for the cache.
	sliced := make([][]float, size0)
	list := make([]float, size0*size1)
	//   CheckAlignment(list)

	for i := 0; i < size0; i++ {
		sliced[i] = list[i*size1 : (i+1)*size1]
	}
	return list, sliced
}

// Allocates a 3D array, as well as the contiguous 1D array backing it.
func Array3D(size0, size1, size2 int) ([]float, [][][]float) {

	// First make the slice and then the list. When the memory is not fragmented,
	// they are probably allocated in a good order for the cache.
	sliced := make([][][]float, size0)
	for i := range sliced {
		sliced[i] = make([][]float, size1)
	}
	list := make([]float, size0*size1*size2)
	//   CheckAlignment(list)

	for i := range sliced {
		for j := range sliced[i] {
			sliced[i][j] = list[(i*size1+j)*size2+0 : (i*size1+j)*size2+size2]
		}
	}
	return list, sliced
}


// Allocates a 4D array, as well as the contiguous 1D array backing it.
func Array4D(size0, size1, size2, size3 int) ([]float, [][][][]float) {

	// First make the slice and then the list. When the memory is not fragmented,
	// they are probably allocated in a good order for the cache.
	sliced := make([][][][]float, size0)
	for i := range sliced {
		sliced[i] = make([][][]float, size1)
	}
	for i := range sliced {
		for j := range sliced[i] {
			sliced[i][j] = make([][]float, size2)
		}
	}
	list := make([]float, size0*size1*size2*size3)
	//   CheckAlignment(list)

	for i := range sliced {
		for j := range sliced[i] {
			for k := range sliced[i][j] {
				sliced[i][j][k] = list[((i*size1+j)*size2+k)*size3+0 : ((i*size1+j)*size2+k)*size3+size3]
			}
		}
	}
	return list, sliced
}

//
func Slice4D(list []float, size []int) [][][][]float {

	sliced := make([][][][]float, size[0])
	for i := range sliced {
		sliced[i] = make([][][]float, size[1])
	}
	for i := range sliced {
		for j := range sliced[i] {
			sliced[i][j] = make([][]float, size[2])
		}
	}

	for i := range sliced {
		for j := range sliced[i] {
			for k := range sliced[i][j] {
				sliced[i][j][k] = list[((i*size[1]+j)*size[2]+k)*size[3]+0 : ((i*size[1]+j)*size[2]+k)*size[3]+size[3]]
			}
		}
	}
	return sliced
}


// Allocates a 4D array, as well as the contiguous 1D array backing it.
func Array5D(size0, size1, size2, size3, size4 int) ([]float, [][][][][]float) {

	// First make the slice and then the list. When the memory is not fragmented,
	// they are probably allocated in a good order for the cache.
	sliced := make([][][][][]float, size0)
	for i := range sliced {
		sliced[i] = make([][][][]float, size1)
	}
	for i := range sliced {
		for j := range sliced[i] {
			sliced[i][j] = make([][][]float, size2)
		}
	}
	for i := range sliced {
		for j := range sliced[i] {
			for k := range sliced[i][j] {
				sliced[i][j][k] = make([][]float, size3)
			}
		}
	}
	list := make([]float, size0*size1*size2*size3*size4)
	//   CheckAlignment(list)

	for i := range sliced {
		for j := range sliced[i] {
			for k := range sliced[i][j] {
				for l := range sliced[i][j][k] {
					sliced[i][j][k][l] = list[(((i*size1+j)*size2+k)*size3+l)*size4+0 : (((i*size1+j)*size2+k)*size3+l)*size4+size4]
				}
			}
		}
	}
	return list, sliced
}


// import (
// 	"reflect"
// )
// 
// // const (
// // 	X = 0
// // 	Y = 1
// // 	Z = 2
// // )
// 
// /** The tensor interface: get size and data */
// 
// type Tensor interface {
// 	Size() []int
// 	Get(index []int) float
// }
// 
// 
// /** Tensor rank = length of size array */
// 
// func Rank(t Tensor) int { return len(t.Size()) }
// 
// /// @deprecated use Len
// func N(t Tensor) int {
// 	n := 1
// 	size := t.Size()
// 	for i := range size {
// 		n *= size[i]
// 	}
// 	return n
// }
// 
// // func Len(t Tensor) int{
// //     n := 1
// //     size := t.Size()
// //     for i := range (size) {
// //         n *= size[i]
// //     }
// //     return n
// // }
// 
// /** Variadic get, utility method. */
// 
// // func Get(t Tensor, index_vararg ... int) float {
// // 	indexarr := ToIntArray(index_vararg)
// // 	return t.Get(indexarr)
// // }
// 
// 
// /** Converts vararg to int array. */
// 
// func ToIntArray(varargs interface{}) []int {
// 	sizestruct := reflect.NewValue(varargs).(*reflect.StructValue)
// 	rank := sizestruct.NumField()
// 	size := make([]int, rank)
// 	for i := 0; i < rank; i++ {
// 		size[i] = sizestruct.Field(i).Interface().(int)
// 	}
// 	return size
// }
// 
// /** Tests Tensor Equality */
// 
// func Equals(dest, source Tensor) bool {
// 	if !EqualSize(dest.Size(), source.Size()) {
// 		return false
// 	}
// 	for i := NewIterator(dest); i.HasNext(); i.Next() {
// 		if dest.Get(i.Index()) != source.Get(i.Index()) {
// 			return false
// 		}
// 	}
// 	return true
// }
// 
// 
// /** Tests if both int slices are equal, in which case they represent equal Tensor sizes. */
// 
// func EqualSize(a, b []int) bool {
// 	if len(a) != len(b) {
// 		return false
// 	} else {
// 		for i := range a {
// 			if a[i] != b[i] {
// 				return false
// 			}
// 		}
// 	}
// 	return true
// }
