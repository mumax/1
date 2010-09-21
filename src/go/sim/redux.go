package sim

import (
	"fmt"
)

type Reductor struct {
	*Backend
	operation          int
	devbuffer          uintptr
	hostbuffer         []float
	blocks, threads, N int
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
	r.Backend = b

	r.threads = 128 //TODO use device default
	for N <= r.threads {
		r.threads /= 2
	}
	r.blocks = divUp(N, r.threads*2)
	r.N = N

	r.devbuffer = b.newArray(r.blocks)
	r.hostbuffer = make([]float, r.blocks)
}

// TODO this should be done by the device code, given the two buffers...
func (r *Reductor) Reduce(data uintptr) float {
	return r.reduce(r.operation, data, r.devbuffer, &(r.hostbuffer[0]), r.blocks, r.threads, r.N)
	// 	r.memcpyFrom(r.devbuffer, &(r.hostbuffer[0]), r.blocks)
	// 	return local_reduce(r.operation, r.hostbuffer)
}

// Integer division but rounded UP
func divUp(x, y int) int {
	return ((x - 1) / y) + 1
}

// Reduce data locally, i.e., not on the GPU.
// This is usually the last step after a partial reduction on the GPU.
// When there are only a few numbers left, it becomes more efficient
// to reduce them on the CPU (we need a copy from the device anyway,
// so why not copy a few numbers).
func local_reduce(operation int, data []float) float {
	fmt.Println(data)
	result := 0.
	switch operation {
	default:
		panic("bug")
	case ADD:
		for i := range data {
			result += data[i]
		}
	case MAX:
		result = data[0]
		for i := range data {
			if result < data[i] {
				result = data[i]
			}
		}
	}
	return result
}
