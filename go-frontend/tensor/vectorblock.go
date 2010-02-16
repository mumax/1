package tensor
// 
// import( 
//   "log";
// );
// 
// 
// /*
//  A vectorblock represents a block of 3-vector data, e.g., the space-dependend magnetization. We store the 3 components of the vector data as 3 continous sub-blocks. I.e., the first 1/3th of the block contains all the x-components, and so on. Such sub-block can be obtained by Block.Component(). E.g.: mx := m.Component(0); returns the space-dependend x-component of m.
// */
// 
// 
// /** Make a new block of 3-vectors. The returned block's first dimension has 3 times the passed size. A component (x, y, z) can be obtained by Block.Component(...) */
// 
// func NewVectorBlock(size []int) *Block{
//   return NewBlock([]int{3*size[0], size[1], size[2]});
// }
// 
// 
// /** Gets a vector component (0, 1 or 2: x, y, or z). The first dimension of the block should be divisible by 3. */
// 
// func (b *Block) Component(c int) *Block{
//   if(c<0 || c>2){
//     log.Crash("Block.Component(): Illegal argument.");
//   }
//   if(b.size[0] % 3 != 0){
//     log.Crash("Block.Component(): Size is not a multiple of 3.");
//   }
//   newsize0 := b.size[0]/3;
//   newsize1 := b.size[1];
//   newsize2 := b.size[2];
//   newvolume := newsize0 * newsize1 * newsize2;
//   slice :=  b.data[c*newvolume:(c+1)*newvolume];
//   // 
//   testAlignment(slice);
//   return &Block{StoredTensor{[]int{newsize0, newsize1, newsize2}, slice}};
// }
// 
// // ugly hack because vectorblok is not 4-dimensional, refactor VectorBlock!?
// 
// func (b *Block) As4Tensor() *VectorBlockTensor{
//   return &VectorBlockTensor{b};
// }
// 
// type VectorBlockTensor struct{
//   block *Block;
// }
// 
// // implements tensor
// func (t *VectorBlockTensor) Size() []int{
//   size := t.block.Component(0).Size();
//   return []int{3, size[0], size[1], size[2]}
// }
// 
// func (t *VectorBlockTensor) Get(index []int) float{
//   return t.block.Component(index[0]).Get3(index[1], index[2], index[3]);
// }
