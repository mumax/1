package tensor

/** An iterator for tensors. */

type Iterator struct{
  tensor Tensor;
  index []int;
  size []int;
  count, max int;
}


func NewIterator(t Tensor) *Iterator{
  return &Iterator{t, make([]int, Rank(t)), t.Size(), 0, N(t)};
}

func (it *Iterator) HasNext() bool{
  return it.count < it.max;
}

func (it *Iterator) Get() float{
  return it.tensor.Get(it.index);
}

func (it *Iterator) Next(){
  it.count++;
  if(it.HasNext()){
    i := len(it.index)-1;
    it.index[i]++;
    for it.index[i] >= it.size[i]{
      it.index[i] = 0;
      i--;
      it.index[i]++;
    }
  }
}

func (it *Iterator) Index() []int{
  return it.index;
}

