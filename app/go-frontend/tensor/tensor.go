package tensor

import (
	. "../assert";
	//. "log";
	//. "fmt";
	"reflect";
	//"io";
	. "math"
)

const (
	X = 0;
	Y = 1;
	Z = 2;
)

/** The tensor interface: get size and data */

type Tensor interface {
	Size() []int;
	Get(index []int) float;
}


/** Tensor rank = length of size array */

func Rank(t Tensor) int { return len(t.Size()) }


func N(t Tensor) int {
	n := 1;
	size := t.Size();
	for i := range (size) {
		n *= size[i]
	}
	return n;
}

/** Variadic get, utility method. */

func Get(t Tensor, index_vararg ...) float {
	indexarr := ToIntArray(index_vararg);
	return t.Get(indexarr);
}


/** Converts vararg to int array. */

func ToIntArray(varargs interface{}) []int {
	sizestruct := reflect.NewValue(varargs).(*reflect.StructValue);
	rank := sizestruct.NumField();
	size := make([]int, rank);
	for i := 0; i < rank; i++ {
		size[i] = sizestruct.Field(i).Interface().(int);
	}
	return size;
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
		for i := range (a) {
			if a[i] != b[i] {
				return false
			}
		}
	}
	return true
}


/** Tensor operations */

/** Slice a tensor by fixing a dimension to value. */

func Slice(t Tensor, dim, value int) *TensorSlice {
	return &TensorSlice{t, dim, value, nil}
}

type TensorSlice struct {
	Tensor;         // the original tensor
	dim, value int; // the dimension to slice away by fixing it to a value
	size       []int;
}

func (t *TensorSlice) Size() []int {
	if t.size == nil {
		origsize := t.Tensor.Size();
		size := make([]int, Rank(t.Tensor)-1);
		for i := 0; i < t.dim; i++ {
			size[i] = origsize[i];
		}
		for i := t.dim + 1; i < len(origsize); i++ {
			size[i-1] = origsize[i];
		}
		t.size = size;
	}
	return t.size;
}

func (t *TensorSlice) Get(index []int) float {

	bigindex := make([]int, Rank(t.Tensor));
	for i := 0; i < t.dim; i++ {
		bigindex[i] = index[i];
	}
	bigindex[t.dim] = t.value;
	for i := t.dim + 1; i < len(bigindex); i++ {
		bigindex[i] = index[i-1];
	}
	return t.Tensor.Get(bigindex);
}

// Note: slice can be use to take a component. //

/** Swap two dimensions. */

func Transpose(t Tensor, x, y int) *TransposedTensor {
	return &TransposedTensor{t, x, y, nil};
}

type TransposedTensor struct {
	Tensor;
	x, y int;
	size []int;
}

func (t *TransposedTensor) Get(index []int) (v float) {
	// swap
	index[t.x], index[t.y] = index[t.y], index[t.x];
	v = t.Tensor.Get(index);
	// swap back
	index[t.x], index[t.y] = index[t.y], index[t.x];
	return;
}

func (t *TransposedTensor) Size() []int {
	if t.size == nil {
		origsize := t.Tensor.Size();
		size := make([]int, len(origsize));
		for i := range (size) {
			size[i] = origsize[i];
		}
		size[t.x], size[t.y] = size[t.y], size[t.x];
		t.size = size;
	}
	return t.size;
}

/** Math */

func Normalize(t Tensor, dim int) *NormalizedTensor {
	return &NormalizedTensor{t, dim};
}

type NormalizedTensor struct {
	Tensor;
	dim int;
}

func (t *NormalizedTensor) Get(index []int) float {
	size := t.Tensor.Size();

	// make an index for going through the direction over which we normalize
	index2 := make([]int, len(size));
	for i := range (index2) {
		index2[i] = index[i];
	}

	// accumulate the total norm of all data along that direction
	var norm2 float64 = 0.;
	for i := 0; i < size[t.dim]; i++ {
		index2[t.dim] = i;
		value := t.Tensor.Get(index2);
		norm2 += float64(value * value);
	}

	return t.Tensor.Get(index) / float(Sqrt(norm2));
}


type TensorSum struct {
	t1, t2 Tensor;
}

func (t *TensorSum) Size() []int { return t.t1.Size() }

func (t *TensorSum) Get(index []int) float { return t.t1.Get(index) + t.t2.Get(index) }

func Add(t1, t2 Tensor) *TensorSum {
	Assert(EqualSize(t1.Size(), t2.Size()));
	return &TensorSum{t1, t2}
}


/** sum of all elements */

type TensorTotal struct {
	original Tensor;
}

func (t *TensorTotal) Size() []int {
	return []int{} // a scalar
}

func (t *TensorTotal) Get(index []int) float {
	Assert(len(index) == 0);

	sum := float64(0.0);

	for it := NewIterator(t); it.HasNext(); it.Next() {
		sum += float64(t.Get(it.Index()))
	}
	return float(sum);
}

func Total(t Tensor) *TensorTotal { return &TensorTotal{t} }
