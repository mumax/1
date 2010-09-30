//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package tensor

/*
 * StoredTensor wraps a multi-dimensional array in the Tensor interface.
 * The array can be passed to FFTW because the underlying storage is contiguous,
 * in contrast to multi-dimensional arrays allocated in the usual way.
 * The contiguous array can be obtained via List().
 *
 * The underlying array is guaranteed to be sufficiently aligned for FFTW.
 */

import (
	"unsafe"
	"log"
)

type StoredTensor interface {
	Size() []int     // Tensor
	Get([]int) float // Tensor
	List() []float   // StoredTensor
}


/** Returns a StoredTensor with given size. For small ranks, specialized implementations like Tensor0, Tensor1, ... are returned. For ranks larger than 5, a general TensorN is returned. */

func NewTensor(size []int) StoredTensor {
	rank := len(size)
	switch {
	case rank == 0:
		return NewTensor0()
	case rank == 1:
		return NewTensor1(size)
	case rank == 2:
		return NewTensor2(size)
	case rank == 3:
		return NewTensor3(size)
	case rank == 4:
		return NewTensor4(size)
	case rank == 5:
		return NewTensor5(size)
	}
	//default:
	return NewTensorN(size)
}


/** Converts multi-dimensional index to integer index in row-major order. Includes range checks. */

func Index(size, indexarray []int) int {

	assertMsg(len(size) == len(indexarray), "Tensor index does not have same rank as tensor size")

	index := indexarray[0]
	assertMsg(!(indexarray[0] < 0 || indexarray[0] >= size[0]), "Index out of range")
	for i := 1; i < len(indexarray); i++ {
		assertMsg(!(indexarray[i] < 0 || indexarray[i] >= size[i]), "Index out of range")
		index *= size[i]
		index += indexarray[i]
	}
	return index
}

// temp hack.
func IndexUnchecked(size, indexarray []int) int {
	//size := t.Size();
	assert(len(size) == len(indexarray))

	index := indexarray[0]
	//assertMsg(! (indexarray[0] < 0 || indexarray[0] >= size[0]), "Index out of range");
	for i := 1; i < len(indexarray); i++ {
		assertMsg(!(indexarray[i] < 0 || indexarray[i] >= size[i]), "Index out of range")
		index *= size[i]
		index += indexarray[i]
	}
	return index
}


// todo: tensor struct should be allocated before arrays

/*
 * 0D stored tensor.
 */

func NewTensor0() *Tensor0 {
	return &Tensor0{[]float{0.}}
}

type Tensor0 struct {
	list []float
}

func (t *Tensor0) Array() float {
	return t.list[0]
}

func (t *Tensor0) List() []float {
	return t.list
}

/** Implements Tensor interface. */

func (t *Tensor0) Get(index []int) float {
	assert(len(index) == 0)
	return t.list[0]
}

func (t *Tensor0) Size() []int {
	return []int{}
}


/*
 * 1D stored tensor.
 */

func NewTensor1(size []int) *Tensor1 {
	return &Tensor1{make([]float, size[0])}
}

type Tensor1 struct {
	list []float
}

func (t *Tensor1) Array() []float {
	return t.list
}

func (t *Tensor1) Component(comp int) *Tensor0 {
	return &Tensor0{t.list[comp:comp]}
}

func (t *Tensor1) List() []float {
	return t.list
}

/** Implements Tensor interface. */

func (t *Tensor1) Get(index []int) float {
	assert(len(index) == 1)
	return t.list[index[0]]
}

func (t *Tensor1) Size() []int {
	return []int{len(t.list)}
}


/*
 * 2D stored tensor.
 */

func NewTensor2(size []int) *Tensor2 {
	list, array := Array2D(size[0], size[1])
	return &Tensor2{array, list}
}

type Tensor2 struct {
	array [][]float
	list  []float
}

func (t *Tensor2) Array() [][]float {
	return t.array
}

func (t *Tensor2) Component(comp int) *Tensor1 {
	return &Tensor1{t.array[comp]}
}

func (t *Tensor2) List() []float {
	return t.list
}

/** Implements Tensor interface. */

func (t *Tensor2) Get(index []int) float {
	assert(len(index) == 2)
	return t.array[index[0]][index[1]]
}

func (t *Tensor2) Size() []int {
	return []int{len(t.array), len(t.array[0])}
}


/*
 * 3D stored tensor.
 */

func NewTensor3(size []int) *Tensor3 {
	list, array := Array3D(size[0], size[1], size[2])
	return &Tensor3{array, list}
}

type Tensor3 struct {
	array [][][]float
	list  []float
}

func (t *Tensor3) Array() [][][]float {
	return t.array
}

func (t *Tensor3) Component(comp int) *Tensor2 {
	start := IndexUnchecked(t.Size(), []int{comp, 0, 0})
	stop := IndexUnchecked(t.Size(), []int{comp + 1, 0, 0})
	return &Tensor2{t.array[comp], t.list[start:stop]}
	// Upper bound is exclusive, so can be out of bounds. That's why we have to use IndexUnchecked().
}

func (t *Tensor3) List() []float {
	return t.list
}

/** Implements Tensor interface. */

func (t *Tensor3) Get(index []int) float {
	assertMsg(len(index) == 3, "Tensor3.Get() needs index of dimension 3")
	return t.array[index[0]][index[1]][index[2]]
}

func (t *Tensor3) Size() []int {
	return []int{len(t.array), len(t.array[0]), len(t.array[0][0])}
}


/*
 * 4D stored tensor.
 */

func NewTensor4(size []int) *Tensor4 {
	list, array := Array4D(size[0], size[1], size[2], size[3])
	return &Tensor4{array, list}
}

type Tensor4 struct {
	array [][][][]float
	list  []float
}

func (t *Tensor4) Array() [][][][]float {
	return t.array
}

func (t *Tensor4) Component(comp int) *Tensor3 {
	start := IndexUnchecked(t.Size(), []int{comp, 0, 0, 0})
	stop := IndexUnchecked(t.Size(), []int{comp + 1, 0, 0, 0})
	return &Tensor3{t.array[comp], t.list[start:stop]}
}

func (t *Tensor4) List() []float {
	return t.list
}

/** Implements Tensor interface. */

func (t *Tensor4) Get(index []int) float {
	assertMsg(len(index) == 4, "Tensor4.Get() needs index of dimension 4")
	return t.array[index[0]][index[1]][index[2]][index[3]]
}

func (t *Tensor4) Size() []int {
	return []int{len(t.array), len(t.array[0]), len(t.array[0][0]), len(t.array[0][0][0])}
}


/*
 * 5D stored tensor.
 */

func NewTensor5(size []int) *Tensor5 {
	list, array := Array5D(size[0], size[1], size[2], size[3], size[4])
	return &Tensor5{array, list}
}

type Tensor5 struct {
	array [][][][][]float
	list  []float
}

func (t *Tensor5) Array() [][][][][]float {
	return t.array
}

func (t *Tensor5) Component(comp int) *Tensor4 {
	start := IndexUnchecked(t.Size(), []int{comp, 0, 0, 0, 0})
	stop := IndexUnchecked(t.Size(), []int{comp + 1, 0, 0, 0, 0})
	return &Tensor4{t.array[comp], t.list[start:stop]}
}

func (t *Tensor5) List() []float {
	return t.list
}

/** Implements Tensor interface. */

func (t *Tensor5) Get(index []int) float {
	assertMsg(len(index) == 5, "Tensor5.Get() needs index of dimension 4")
	return t.array[index[0]][index[1]][index[2]][index[3]][index[4]]
}

func (t *Tensor5) Size() []int {
	return []int{len(t.array), len(t.array[0]), len(t.array[0][0]), len(t.array[0][0][0]), len(t.array[0][0][0][0])}
}


/*
 * N-Dimensional StoredTensor
 */

func NewTensorN(size []int) *TensorN {
	//size := ToIntArray(dimensions);  // todo: copy size
	rank := len(size)

	n := 1
	for i := 0; i < rank; i++ {
		n *= size[i]
	}

	t := TensorN{size, make([]float, n)}
	return &t
}

type TensorN struct {
	size []int   // size in x, y and z dimensions
	list []float // data as a continous array
}

/** Implements Tensor */
//todo: return copy of size
func (t *TensorN) Size() []int {
	return t.size
}

func (t *TensorN) Get(index []int) float {
	i := Index(t.Size(), index)
	return t.list[i]
}

/** Implements StoredTensor */

func (t *TensorN) List() []float {
	return t.list
}


func (t *TensorN) Component(comp int) *TensorN {
	index := make([]int, len(t.size))
	index[0] = comp
	start := IndexUnchecked(t.size, index)
	index[0] = comp + 1
	stop := IndexUnchecked(t.size, index)
	size := make([]int, len(t.size)-1)
	for i := range size {
		size[i] = t.size[i+1]
	}
	return &TensorN{size, t.list[start:stop]}
}


/*
 * Methods for all stored tensors
 */


/** Pointer to the 0th element of the data array, for use with C code, e.g. */

func DataAddress(t StoredTensor) unsafe.Pointer {
	return unsafe.Pointer(&(t.List())[0])
}

/** Set element */

func Set(t StoredTensor, index []int, value float) {
	i := Index(t.Size(), index)
	t.List()[i] = value
}


/** Set all elements to value */

func SetAll(t StoredTensor, value float) {
	list := t.List()
	for i := range list {
		list[i] = value
	}
}


/** Fills tensor with zeros. */

func Zero(t StoredTensor) {
	SetAll(t, 0.)
}


/** Copies data from a StoredTensor to an existing StoredTensor. */

func CopyTo(source, dest StoredTensor) {
	assert(EqualSize(dest.Size(), source.Size()))
	d, s := dest.List(), source.List()
	for i := range d {
		d[i] = s[i]
	}
}


/** Makes a fresh copy of a StoredTensor. */

func Copy(t StoredTensor) StoredTensor {
	clone := NewTensor(t.Size())
	CopyTo(t, clone)
	return clone
}


/** Copies a Tensor into a freshly made StoredTensor. */

func BufferTo(source Tensor, dest StoredTensor) {
	assert(EqualSize(dest.Size(), source.Size()))
	for i := NewIterator(dest); i.HasNext(); i.Next() {
		Set(dest, i.Index(), source.Get(i.Index()))
	}
}


/** Copies a Tensor into a freshly made StoredTensor. */

func Buffer(t Tensor) StoredTensor {
	clone := NewTensor(t.Size())
	BufferTo(t, clone)
	return clone
}


/*
 * Private
 */


/** We mimic fftw's malloc here, so that the array is aligned for SIMD instructions. */
func mallocAligned(size int) []float {
	log.Stderr("mallocAligned(", size, ")")
	array := make([]float, size)
	// better way: make array a bit too big and slice'em!, perhaps make it 32-bite aligned, to be sure.
	CheckAlignment(array)
	return array
}

func CheckAlignment(array []float) {
	i := ToInt(unsafe.Pointer(&array[0])) // replace by DataAddress;
	if i%16 != 0 {
		log.Crash("Misalignment, sorry...")
	}
}
