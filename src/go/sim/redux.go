//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import (
)


type Reductor struct {
	*Backend
	operation          int
	devbuffer          uintptr
	hostbuffer         []float32
	blocks, threads, N int
}


// Reduces the data,
// i.e., calucates the sum, maximum, ...
// depending on the value of "operation".
func (r *Reductor) Reduce(input *DevTensor) float32 {
	assert(prod(input.size) == r.N)
	return r.reduce(r.operation, input.data, r.devbuffer, &(r.hostbuffer[0]), r.blocks, r.threads, r.N)
}


// Unsafe version of Reduce().
func (r *Reductor) reduce_(data uintptr) float32 {
	return r.reduce(r.operation, data, r.devbuffer, &(r.hostbuffer[0]), r.blocks, r.threads, r.N)
}


func NewSum(b *Backend, N int) *Reductor {
	r := new(Reductor)
	r.InitSum(b, N)
	return r
}


func (r *Reductor) InitSum(b *Backend, N int) {
	r.init(b, N)
	r.operation = ADD
}


func NewMax(b *Backend, N int) *Reductor {
	r := new(Reductor)
	r.InitMax(b, N)
	return r
}


func (r *Reductor) InitMax(b *Backend, N int) {
	r.init(b, N)
	r.operation = MAX
}


func NewMin(b *Backend, N int) *Reductor {
	r := new(Reductor)
	r.InitMin(b, N)
	return r
}


func (r *Reductor) InitMin(b *Backend, N int) {
	r.init(b, N)
	r.operation = MIN
}


func NewMaxAbs(b *Backend, N int) *Reductor {
	r := new(Reductor)
	r.InitMaxAbs(b, N)
	return r
}

func (r *Reductor) InitMaxAbs(b *Backend, N int) {
	r.init(b, N)
	r.operation = MAXABS
}


// initiates the common pieces of all reductors
func (r *Reductor) init(b *Backend, N int) {
	assert(N > 1)
	assert(b != nil)
	r.Backend = b

	r.threads = b.maxthreads() / 2 // does not work with maxthreads
	for N <= r.threads {
		r.threads /= 2
	}
	r.blocks = divUp(N, r.threads*2)
	r.N = N

	r.devbuffer = b.newArray(r.blocks)
	r.hostbuffer = make([]float32, r.blocks)
}


// Integer division but rounded UP
func divUp(x, y int) int {
	return ((x - 1) / y) + 1
}

// OBSOLETE now done in C code.
// Reduce data locally, i.e., not on the GPU.
// This is usually the last step after a partial reduction on the GPU.
// When there are only a few numbers left, it becomes more efficient
// to reduce them on the CPU (we need a copy from the device anyway,
// so why not copy a few numbers).
// func local_reduce(operation int, data []float32) float32 {
// 	fmt.Println(data)
// 	result := float32(0.)
// 	switch operation {
// 	default:
// 		panic("bug")
// 	case ADD:
// 		for i := range data {
// 			result += data[i]
// 		}
// 	case MAX:
// 		result = data[0]
// 		for i := range data {
// 			if result < data[i] {
// 				result = data[i]
// 			}
// 		}
// 	}
// 	return result
// }
