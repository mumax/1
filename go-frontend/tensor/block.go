 package tensor
// 
// import( "unsafe";
// 	"../libsim";
// 	//. "../assert";
// 	"log";
// 	"fmt";
// 	"io";
// 	. "os";
// );
// 
// 
// /** A 3-dimensional block of data, backed by a general tensor but limited to 3 dimensions and guaranteed to be aligned for use with fftw3. */
// 
// type Block struct{
//   StoredTensor;
// }
// 
// 
// /** Returns the size of dimension 0 (x) */
// 
// func (b *Block) Size0() int{
//   return b.size[0];
// }
// 
// 
// /** Returns the size of dimension 1 (y) */
// 
// func (b *Block) Size1() int{
//   return b.size[1];
// }
// 
// 
// /** Returns the size of dimension 2 (z) */
// 
// func (b *Block) Size2() int{
//   return b.size[2];
// }
// 
// 
// /** Make a new block with given Size */
// 
// func NewBlock(size []int) *Block{
//   // Data should be nicely aligned for fftw
//   data := mallocAligned(Product(size));
//   // size slice gets copied so it can not be changed later
//   return &Block{StoredTensor{[]int{size[0], size[1], size[2]}, data}};
// }
// 
// 
// 
// 
// 
// 
// /***********************************************************
//  * Access/copying
//  ***********************************************************/
// 
// 
// /** Implements Tensor. */
// 
// func (b *Block) Get(index []int) float{
//   return b.StoredTensor.Get(index);
// }
// 
// 
// /** Get element i,j,k */
// 
// func (b *Block) Get3(i, j, k int) float{
//   // supercall:
//   return b.StoredTensor.Get([]int{i,j,k});
// 
//   // older code, works OK though:
//   /*
//   if i<0 || j<0 || k<0 || i>= b.size[0] || j>= b.size[1] || k >= b.size[2]{
//     util.ErrorMsg("Set: block index out of bounds: " + I(i) + ", " + I(j) + ", " + I(k));
//   }
//   //return libsim.BlockGet(b.size[0], b.size[1], b.size[2], b.datapointer, i, j, k);
//   return b.data[ b.size[2]*(i*b.size[1] + j) + k ];*/
// }
// 
// 
// /** Get element i,j,k without bound checks of x,y,z sizes, only total array size*/
// 
// func (b *Block) GetUnchecked(i, j, k int) float{
//   // for now we DO check, will remove check later.
//   return b.Get3(i,j,k);
// }
// 
// 
// /** Set element i,j,k to value */
// 
// func (b *Block) Set(i, j, k int, value float){
//   b.StoredTensor.Set([]int{i, j, k}, value);
// 
//   // old but OK code:
//   /*
//   if i<0 || j<0 || k<0 || i>= b.size[0] || j>= b.size[1] || k >= b.size[2]{
//     util.ErrorMsg("Set: block index out of bounds: " + I(i) + ", " + I(j) + ", " + I(k));
//   }
//     b.data[ b.size[2]*(i*b.size[1] + j) + k ] = value;*/
// }
// 
// 
// /** Set element i,j,k to value without bound checks of x,y,z sizes, only total array size*/
// 
// func (b *Block) SetUnchecked(i, j, k int, value float){
//     b.Set(i, j, k, value);
// }
// 
// 
// 
// 
// /** Copy to a block with equal Size */
// 
// func Copy(source, dest *Block){
//   if EqualSize(source.size[0:3], dest.size[0:3]){
//     for i:= range(source.data){
//       dest.data[i] = source.data[i];
//     }
//   }
//   else{
//     log.Crash("Copy: size mismatch");
//   }
// }
// 
// func (dest *Block) SetTo(source *Block){
//   Copy(source, dest);
// }
// 
// 
// func (source *Block) CopyTo(dest *Block){
//   Copy(source, dest);
// }
// 
// /** Copy into a block with equal or larger Size. Unused elements in the larger block are not zeroed */
// 
// func (source *Block) CopyInto(dest *Block){
//   if LargerSize(source.size[0:3], dest.size[0:3]){
//     log.Crash("CopyInto: source is larger than dest");
//   }
//   else{
//     for i:= 0; i<source.size[0]; i++{
//       for j:= 0; j<source.size[1]; j++{
// 	for k:= 0; k<source.size[2]; k++{
// 	  dest.Set(i,j,k, source.Get3(i, j, k));	// can be sped up quite a bit.
// 	}
//       }
//     }
//   }
// }
// 
// 
// /** Copy from a block with equal or larger Size. Unused elements in the larger block are ignored */
// 
// func (dest *Block) CopyFrom(source *Block){
//   if LargerSize(dest.size[0:3], source.size[0:3]){
//     log.Crash("CopyFrom: dest is larger than source");
//   }
//   else{
//     for i:= 0; i<dest.size[0]; i++{
//       for j:= 0; j<dest.size[1]; j++{
// 	for k:= 0; k<dest.size[2]; k++{
// 	  dest.Set(i,j,k, source.Get3(i, j, k));	// can be sped up quite a bit.
// 	}
//       }
//     }
//   }
// }
// 
// 
// 
// 
// /** Allocates a new Block, copy of the original */
// 
// func (orig *Block) Clone() *Block{
//   clone := NewBlock(orig.Size());
//   clone.SetTo(orig);
//   return clone;
// }
// 
// 
// /** Test if sizes and all elements are equal */
// 
// func (a *Block) Equals(b *Block) bool{
//   for i:= range(a.size){
//     if (a.size[i] != b.size[i]){
//       return false;
//     }
//   }
// 
//   for i:= range(a.data){
//     if(a.data[i] != b.data[i]){
//       return false;
//     }
//   }
// 
//   return true;
// }
// 
// /*****************************************************************
//  * I/0
//  *****************************************************************/
// 
// /** Print the block */
// func (b *Block) Print(out io.Writer){
//   PrintTensor(out, &b.StoredTensor);
// }
// 
// /****************************************************************
//  * Utilities
//  ****************************************************************/
// 
// /** Product of the elements of an N-dimensional Size gives its total Size (number of elements) */
// func Product(array []int) int{
//   product := 1;
//   for i:=range(array){
//     product *= array[i];
//   }
//   return product;
// }
// 
// /** Tests if two things have the same Size */
// 
// 
// /** Tests if a is larger than b, in any dimension */
// func LargerSize(a, b []int) bool{
//   if len(a) == len(b){
//     for i:=range(a){
//       if(a[i] <= b[i]){
// 	return false;
//       }
//     }
//   }
//   else{
//     fmt.Fprintf(Stderr, "Dimension mismatch: ", a , " ", b);
//     Exit(1);
//   }
//   return true;
// }
// 
// func I(i int) string{
//   return fmt.Sprintf("%d", i);
// }
