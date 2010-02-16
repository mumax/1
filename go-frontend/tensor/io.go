package tensor

import(
  . "../assert";
  . "fmt";
  "io";
  "os";
  "log";
)

func PrintTensor(out io.Writer, t Tensor){
//   rank := Rank(t);
//   
//   if rank == 0{
//     Fprintf(out, "% f", Get(t), "\n");
//     return;
//   }
//   
//   if rank <= 3 {
//     size := [3]int{};
//     // promote to a rank 3 tensor of size 1x1xN or 1xNxM
//     for i:=range(size){
//       if rank < i+1{
// 	size[i] = 1;
//       }
//       else {
// 	size[i] = t.Size()[i];
//       }
//     }
// 
//     for k:=0; k<size[2]; k++ {
//       for j:=0; j<size[1]; j++ {
// 	for i:=0; i<size[0]; i++ {
// 	  Fprintf(out, "% f ", Get(t, i, j, k));
// 	}
// 	Fprintf(out, "\n");
//       }
//       Fprintf(out, "\n");
//     }
//     return;
//   }
// 
//   //else{
//   //  Fprintf(out, "Can not yet print tensors with rank > 3");
//   //}   
}


/** Prints an unstructured field of vectors (3 co-ordinates and 3 vector components per line), suitable for Gnuplot 'plot with vectors' */

func PrintVectors(out io.Writer, t Tensor){
  AssertMsg(t.Size()[0]==3, "Needs first dimension of size 3 (vector components)");
  xcomp := Slice(t, 0, X);
  ycomp := Slice(t, 0, Y);
  zcomp := Slice(t, 0, Z);

  for it := NewIterator(xcomp); it.HasNext(); it.Next(){
    index := it.Index();
    for i:=0; i<len(it.Index()); i++{
      Fprintf(out, "% d", index[i]);
    }
    Fprint(out, " ", xcomp.Get(index), ycomp.Get(index), zcomp.Get(index), "\n");
  }

  // close if possible.
  c:=out.(io.Closer);
  if  c != nil{
    out.(io.Closer).Close();
  }
}




/* Todo: sometimes appends instead of overwriting...
  move to util?
 */

func FOpen(filename string) io.Writer{
  file, ok := os.Open(filename, os.O_WRONLY | os.O_CREATE | os.O_TRUNC, 0666);
  if ok!=nil{
    Fprint(os.Stderr, ok, "\n");
    log.Crash("Could not open file");
  }
  return file;
}
