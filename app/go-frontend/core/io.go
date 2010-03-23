package core

import(
  . "fmt";
  "io";
  "os";
  "log";
  "libsim";
)

// TODO(general): refactor all names to go convention: tensor.Print(), sim.New(), assert.Check(), ... 
// use "." imports a bit less.


/** Writes the tensor in 32-bit binary format. See libtensor.h for more details. */
func Write(out io.Writer, t Tensor){
  //Fprintln(os.Stderr, "WriteTensor(", &t, out, ")");
  buffer := make([]byte, 4);
  
  libsim.IntToBytes(Rank(t), buffer);	// probably could have used unformatted printf, scanf here?
  _, err := out.Write(buffer);
  if err != nil{ log.Crash(err) }
  
  for i:=range(t.Size()){
      libsim.IntToBytes(t.Size()[i], buffer);
      _, err := out.Write(buffer);
      if err != nil{ log.Crash(err) }
  }
  
  for i:=NewIterator(t); i.HasNext(); i.Next(){
      libsim.FloatToBytes(i.Get(), buffer);
      _, err := out.Write(buffer);
      if err != nil{ log.Crash(err) }
  }
}

/** Reads a tensor from 32-bit binary format. See libtensor.h for more details. */
func Read(in io.Reader) StoredTensor{
  //Fprintln(os.Stderr, "ReadTensor(", in, ")");
  buffer := make([]byte, 4);
  
  in.Read(buffer);
  rank := libsim.BytesToInt(buffer);
  
  size := make([]int, rank);
  for i:=range(size){
    in.Read(buffer);
    size[i] = libsim.BytesToInt(buffer);
  }
  
  t := NewTensorN(size);
  list := t.List();
  
  for i:=range(list){
    in.Read(buffer);
    list[i] = libsim.BytesToFloat(buffer);
  }
  
  return t;
}

/** Prints the tensor in ASCII format, See libtensor.h for more details. */
func Print(out io.Writer, t Tensor){
  Fprintln(out, Rank(t));
  
  for i:=range(t.Size()){
    Fprintln(out, t.Size()[i]);
  }
  
  for i:=NewIterator(t); i.HasNext(); i.Next(){
    Fprintln(out, i.Get());
  }
}

/** Prints the tensor in ASCII with some row/column formatting to make it easier to read for humans. */
func Format(out io.Writer, t Tensor){
  Fprintln(out, Rank(t));
  
  for i:=range(t.Size()){
    Fprintln(out, t.Size()[i]);
  }
  
  for i:=NewIterator(t); i.HasNext(); i.Next(){
    Fprintln(out, i.Get());
  }
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
  TODO: remove the duplicate in util
 */

func FOpen(filename string) io.Writer{
  file, ok := os.Open(filename, os.O_WRONLY | os.O_CREATE | os.O_TRUNC, 0666);
  if ok!=nil{
    Fprint(os.Stderr, ok, "\n");
    log.Crash("Could not open file");
  }
  return file;
}
