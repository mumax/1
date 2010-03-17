package main
  
import( 
       "flag";
       "fmt";
       . "../tensor";
       "os";
       . "../util";
       "strconv";
)

var ascii = flag.Bool("ascii", false, "output in ASCII format rather than binary")
var gnuplot = flag.Bool("gnuplot", false, "print in a format usable for gnuplot splot with vectors")

//var format = flag.Bool("format", false, "if outputting ASCII, format it in rows and columns")
var component = flag.String("component", "", "select a component")


func main() {
  flag.Parse();
  
  for i:=0; i<flag.NArg(); i++{
    var tensor Tensor = Read(FOpenz(flag.Arg(i)));
    
    if len(*component) != 0{
      tensor = Slice(tensor, 0, Atoi(*component));
    }
    
    if *gnuplot{
      PrintVectors(os.Stdout, tensor);
      return;
    }
    
    
    if *ascii{
      Print(os.Stdout, tensor);
    } else {
      Write(os.Stdout, tensor);
    }
  }
  
}


func Atoi(s string) int{
  // idea: also parse x, y, z to 0, 1, 2
  i, err := strconv.Atoi(s);
  if err != nil { 
    fmt.Fprintln(os.Stderr, "Expecting integer argument:", err);
    os.Exit(-1);
  }
  return i;
}
