package main
  
import( 
       "fmt";
       . "../tensor";
       "os";
)

var tensor Tensor;

func main() {
  commands, args := parseArgs();
  for c:=range(commands){
    exec(commands[c], args[c]);
  }
}

// BUG: reading from stdin hangs...

func exec(command string, args []string){
  switch command{
    case "--read":
      tensor = Read(parseFileOrStdout(args));
    case "--new":
      rank := len(args);
      size := make([]int, rank);
      for i:=range(size){ size[i] = Atoi(args[i]) }
      tensor = NewTensorN(size);

    case "--rank":
      fmt.Fprintln(parseFileOrStdout(args), Rank(tensor));
    case "--size":
      out := parseFileOrStdout(args);
      for i:= range(tensor.Size()){ fmt.Fprintln(out, tensor.Size()[i]) };

    case "--component":
      argCount(command, args, 1, 1);
      tensor = Slice(tensor, 0, Atoi(args[0]));
    case "--slice":
      argCount(command, args, 2, 2);
      tensor = Slice(tensor, Atoi(args[0]), Atoi(args[1]));

    case "--normalize":
      argCount(command, args, 1, 1);
      tensor = Normalize(tensor, Atoi(args[0]));

    case "--set":
      argCount(command, args, Rank(tensor)+1, Rank(tensor)+1);
      t := Buffer(tensor);
      index := make([]int, Rank(tensor));
      for i:=0; i<len(args)-1; i++ { index[i] = Atoi(args[i]) }
      Set(t, index, Atof(args[len(args)-1]));
      tensor = t;
    case "--setall":
      argCount(command, args, 1, 1);
      t := Buffer(tensor);
      SetAll(t, Atof(args[0]));
      tensor = t;

    case "--format":
      Format(parseFileOrStdout(args), tensor);
    case "--write":
      Write(parseFileOrStdout(args), tensor);
    case "--gnuplot":
      PrintVectors(parseFileOrStdout(args), tensor);

    default:
      fmt.Fprintln(os.Stderr, "unknown command:", command);
      os.Exit(-1);
  }
  
}

