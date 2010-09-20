package sim


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
