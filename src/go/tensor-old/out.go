//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package tensor

import (
	"fmt"
	//"io";
	"os"
	"log"
)

// idea: merge with assert(?)


// TODO: rename function, FOpen already exists.
/** Todo: sometimes appends instead of overwriting... */
func FOpenz(filename string) *os.File {
	file, ok := os.Open(filename, os.O_RDWR|os.O_CREAT, 0666)
	if ok != nil {
		fmt.Fprint(os.Stderr, ok, "\n")
		log.Crash("Could not open file")
	}
	return file
}


/** Prints an unstructured field of vectors (3 co-ordinates and 3 vector components per line), suitable for Gnuplot 'plot with vectors' */
/*
func PrintVectors(out io.Writer, t Tensor){
  Assert(Rank(t)==4, "ToGnuplotVectors: needs 4D data");
  Assert(t.Size()[0]==3, "ToGnuplotVectors: needs first dimension of size 3 (vector components)");
  xcomp := Slice(t, 0, X);
  ycomp := Slice(t, 0, Y);
  zcomp := Slice(t, 0, Z);

  for it := NewIterator(xcomp); it.HasNext(); it.Next(){
    index := it.Index();
    for i:=0; i<3; i++{
      fmt.Fprintf(out, "% d", index[i]);
    }
    fmt.Fprint(out, " ", xcomp.Get(index), ycomp.Get(index), zcomp.Get(index), "\n");
  }

  // close if possible.
  c:=out.(io.Closer);
  if  c != nil{
    out.(io.Closer).Close();
  }
}*/

/** ugly temp. hack, make general: Vectorblock is really a 4D stored tensor. */

// func PrintVectorBlock(out io.Writer, t *Block){
//
//   xcomp := t.Component(X);
//   ycomp := t.Component(Y);
//   zcomp := t.Component(Z);
//
//   for it := NewIterator(xcomp); it.HasNext(); it.Next(){
//     index := it.Index();
//     for i:=0; i<3; i++{
//       fmt.Fprintf(out, "% d", index[i]);
//     }
//     fmt.Fprint(out, " ", xcomp.Get(index), ycomp.Get(index), zcomp.Get(index), "\n");
//   }
//
//   // close if possible.
//   c:=out.(io.Closer);
//   if  c != nil{
//     out.(io.Closer).Close();
//   }
// }
