package sim

import(
  "unsafe"
  )
  
type Reductor struct{
  *Backend
  operation int
  devbuffer unsafe.Pointer
  hostbuffer []float
  blocks, threads, N int
}

func NewSum(b *Backend, N int) *Reductor{
    r := new(Reductor)
    r.InitSum(b, N)
    return r
}

func (r *Reductor) InitSum(b *Backend, N int){
  r.Backend = b
  r.operation = ADD

  r.threads = 128 //TODO use device default
  for N <= r.threads{
    r.threads /= 2
  }
  r.blocks = divUp(N, r.threads*2)
  r.N = N

  r.devbuffer = b.newArray(r.blocks)
  r.hostbuffer = make([]float, r.blocks)
}

// TODO this should be done by the device code, given the two buffers...
func (r *Reductor) Reduce(data unsafe.Pointer) float{
   r.reduce(r.operation, data, r.devbuffer, r.blocks, r.threads, r.N)
   r.memcpyFrom(r.devbuffer, &(r.hostbuffer[0]), r.blocks)
   sum := 0.
   for i:=range(r.hostbuffer){
      sum += r.hostbuffer[i]
   }
   return sum
}

// Integer division but rounded UP
func divUp(x, y int) int{
  return  ((x-1)/y) +1
}

// func (d Gpu)  reduce(operation int, input, output unsafe.Pointer, blocks, threads, N int){
//   C.gpu_reduce(C.int(operation), (*C.float)(input), (*C.float)(output), C.int(blocks),  C.int(threads),  C.int(N))
// }

// ///@ todo leaks memory, should not allocate
// float gpu_sum(float* data, int N){
// 
//   int threads = 128;
//   while (N <= threads){
//     threads /= 2;
//   }
//   int blocks = divUp(N, threads*2);
// 
//   float* dev2 = new_gpu_array(blocks);
//   float* host2 = new float[blocks];
// 
//   gpu_partial_sums(data, dev2, blocks, threads, N);
// 
//   memcpy_from_gpu(dev2, host2, blocks);
// 
//   float sum = 0.;
// 
//   for(int i=0; i<blocks; i++){
//     sum += host2[i];
//   }
//   //gpu_free(dev2);
//   //delete[] host2;
//   return sum;
// }
