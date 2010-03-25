package main
  
import( 
  . "../tensor";
  . "../sim";
  "os";
  "fmt";
)


func main() {
  commands, args := parseArgs();
  for i:=range(commands){
    exec(commands[i], args[i]);
  }
  if kernel != nil{ Write(os.Stdout, kernel) } else { fmt.Fprintln(os.Stderr, "no kernel generated.")}
}

var kernel Tensor;

func exec(command string, args []string){
  switch command{
    case "--unit":
      kernel = UnitKernel(parseSize(args));
    case "--dipole":
      kernel = DipoleKernel(parseSize(args));
    case "--cubic":
      kernel = PointKernel(parseSize(args), 1.);
    default:
      fmt.Fprintln(os.Stderr, "unknown command:", command);
      os.Exit(-1);
  }
  
}

func parseSize(args []string) []int{
  argCount("size", args, 3, 3);
  size := make([]int, 3);
  for i:=0; i<3; i++{
    size[i] = Atoi(args[i]);
  }
  return size;
}

