package refsh

import(
  . "reflect";
   "fmt";
   "os";
   "strconv";
)

/** 
 * Maximum number of functions.
 * @todo use a vector to make this unlimited.
 */
const CAPACITY = 100;


type Refsh struct{
  funcnames []string;
  funcs []*FuncValue;
}

func NewRefsh() *Refsh{
  return &Refsh{make([]string, CAPACITY)[0:0], make([]*FuncValue, CAPACITY)[0:0]}; 
}

func New() *Refsh{
  return NewRefsh();
}

/**
 * Adds a function to the list of known commands.
 * example: refsh.Add("exit", reflect.NewValue(Exit));
 */
func (r *Refsh) Add(funcname string, function Value){
  if r.resolve(funcname) != nil{
    fmt.Fprintln(os.Stderr, "Aldready defined:", funcname);
    os.Exit(-4);
  }
  r.funcnames = r.funcnames[0:len(r.funcnames)+1];
  r.funcnames[len(r.funcnames)-1] = funcname;
  r.funcs = r.funcs[0:len(r.funcs)+1];
  r.funcs[len(r.funcs)-1] = function.(*FuncValue);
}

/**
 * Calls a function. Function name and arguments are passed as strings.
 * The function name should first have been added by refsh.Add();
 */
func (refsh *Refsh) Call(fname string, argv []string){
  function := refsh.resolve(fname);
  if function == nil{
    fmt.Fprintln(os.Stderr, "Unknown command:", fname, "Options are:", refsh.funcnames);
    os.Exit(-5);
  }

  args := refsh.parseArgs(fname, argv);
  function.Call(args);
}

/**
 * Executes the command line arguments. They should have a syntax like:
 * --command1 arg1 arg2 --command2 --command3 arg1
 */
func (refsh *Refsh) ExecFlags(){
  commands, args := ParseFlags();
  for i:=range(commands){
    //fmt.Fprintln(os.Stderr, commands[i], args[i]);
    refsh.Call(commands[i], args[i]);
  }
}



func (r *Refsh) resolve(funcname string) *FuncValue{
  for i:=range(r.funcnames){
    if r.funcnames[i] == funcname{
      return r.funcs[i];
    }
  }
  return nil; // never reached
}

func (refsh *Refsh) parseArgs(fname string, argv []string) []Value{
  function := refsh.resolve(fname);
  functype := function.Type().(*FuncType);
  nargs := functype.NumIn();

  checkArgCount(fname, argv, nargs);

  args := make([]Value, nargs);
  for i:=range(args){
    args[i] = parseArg(argv[i], functype.In(i));
  }
  return args;
}

func checkArgCount(fname string, argv []string, nargs int){
  if nargs != len(argv){
    fmt.Fprintln(os.Stderr, "Error calling", fname, argv, ": needs", nargs, "arguments.");
    os.Exit(-1);
  }
}

func parseArg(arg string, argtype Type) Value{
  switch argtype.Name(){
    default: 
      fmt.Fprintln(os.Stderr, "Do not know how to parse", argtype);
      os.Exit(-2);
    case "int":
      return NewValue(parseInt(arg));
  }
  return NewValue(666); // is never reached.
}

func parseInt(str string) int{
  i, err := strconv.Atoi(str);
  if err != nil{
      fmt.Fprintln(os.Stderr, "Could not parse to int:", str);
      os.Exit(-3);
  }
  return i;
}


func main(){
  refsh := NewRefsh();
  refsh.Add("test", NewValue(Test));
  refsh.ExecFlags();
}

func Test(i int){
  fmt.Println("Hello reflection!", i);
}



