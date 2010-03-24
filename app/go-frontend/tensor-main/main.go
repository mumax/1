package main
  
import( 
       "flag";
       "fmt";
       . "../tensor";
       "os";
       "strconv";
       "strings";
       "log";
)

var tensor Tensor;

func main() {
  commands, args := parseArgs();
  for c:=range(commands){
    exec(commands[c], args[c]);
  }
}

func exec(command string, args []string){
  fmt.Println(command, args);
  switch command{
    case "--read":
      argCount(command, args, 1, 1);
      tensor = ReadFile(args[0]);
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

func parseFileOrStdout(args []string) *os.File{
  argCount("an output function", args, 0, 1);
  if len(args) == 0 { return os.Stdout}
  return FOpenz(args[0]);
}

func argCount(command string, args []string, min, max int){
  if len(args) < min || len(args) > max{
    if min != max { 
      fmt.Fprintln(os.Stderr, command, "expects", min, "to", max, "parameters") 
    } else {
      fmt.Fprintln(os.Stderr, command, "expects", min, "parameters") 
    }
    os.Exit(-1);
  }
}

func parseArgs() ([]string, [][]string){
  ncommands := 0;
  for i:=0; i<flag.NArg(); i++{
      if strings.HasPrefix(flag.Arg(i), "--"){
	ncommands++;
      }
  }
  
  commands := make([]string, ncommands);
  args := make([][]string, ncommands);

  {command := 0;
  i:=0;
  for i < flag.NArg(){
    assert(strings.HasPrefix(flag.Arg(i), "--"));
    commands[command] = flag.Arg(i);
    nargs := 0;
    i++;
    for i < flag.NArg() && !strings.HasPrefix(flag.Arg(i), "--"){
      nargs++;
      i++;
    }
    args[command] = make([]string, nargs);
    command++;
  }}

  {command := 0;
  i:=0;
   for i < flag.NArg(){
    assert(strings.HasPrefix(flag.Arg(i), "--"));
    commands[command] = flag.Arg(i);
    nargs := 0;
    i++;
    for i < flag.NArg() && !strings.HasPrefix(flag.Arg(i), "--"){
      args[command][nargs] = flag.Arg(i);
      nargs++;
      i++;
    }
    command++;
  }}
  return commands, args;
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

func Atof(s string) float{
  f, err := strconv.Atof(s);
  if err != nil { 
    fmt.Fprintln(os.Stderr, "Expecting floating-point argument:", err);
    os.Exit(-1);
  }
  return f;
}


/** Crashes the program when the test is false. */
func assert(test bool){
  if !test{
    log.Crash("Assertion failed");
  }
}

/** Crashes the program with an error message when the test is false. */
func assertMsg(test bool, msg string){
  if !test{
    log.Crash(msg);
  }
}

