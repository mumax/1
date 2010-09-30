//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package tensor

/**
 * An iterator for tensors.
 *
 * Handy to access a tensor when the rank is not known and the tensor is
 * not necessarily a StoredTensor (so there is no .List() ).
 * Iterator.Next() iterates to the next element, in row-major oder.
 *
 * Typical usage:
 *
 * for i := NewIterator(tensor); i.HasNext(); i.Next(){
 *    element := i.Get();
 *    current_position = i.Index();
 * }
 *
 */

type Iterator struct {
	tensor     Tensor
	index      []int
	size       []int
	count, max int
}

/**
 * New iterator for the tensor,
 * starts at 0th element and can not be re-used.
 */
func NewIterator(t Tensor) *Iterator {
	return &Iterator{t, make([]int, Rank(t)), t.Size(), 0, N(t)}
}

func (it *Iterator) HasNext() bool {
	return it.count < it.max
}

func (it *Iterator) Get() float {
	return it.tensor.Get(it.index)
}

func (it *Iterator) Next() {
	it.count++
	if it.HasNext() {
		i := len(it.index) - 1
		it.index[i]++
		for it.index[i] >= it.size[i] {
			it.index[i] = 0
			i--
			it.index[i]++
		}
	}
}

/** Returns the current N-dimensional index. */

func (it *Iterator) Index() []int {
	return it.index
}
