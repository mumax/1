package main
  
import( 
       "flag";
       "fmt";
       "os";
       "strconv";
       "strings";
       "log";
)

func parseFileOrStdout(args []string) *os.File{
  argCount("an output function", args, 0, 1);
  if len(args) == 0 { return os.Stdout}
  return fOpenz(args[0]);
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

// TODO: rename function, FOpen already exists.
/** Todo: sometimes appends instead of overwriting... */
func fOpenz(filename string) *os.File{
  file, ok := os.Open(filename, os.O_RDWR | os.O_CREAT, 0666);
  if ok!=nil{
    fmt.Fprint(os.Stderr, ok, "\n");
    log.Crash("Could not open file");
  }
  return file;
}

