package tensor

type Resampled struct {
	original Tensor
	size     []int
}

func (t *Resampled) Size() []int {
	return t.size
}

func (t *Resampled) Get(index []int) float {
	index2 := make([]int, len(index))
	for i := range index {
		index2[i] = (index[i] * t.size[i]) / t.original.Size()[i]
	}
	return t.original.Get(index2)
}

func Resample(t Tensor, size []int) Tensor {
	return &Resampled{t, size}
}
