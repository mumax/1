package main

import(
  . "reflect";
  "fmt";
  "os";
  "strconv";
)

type Refsh struct{
  Funcs map[string]*FuncValue;
}

func NewRefsh() *Refsh{
  return &Refsh{map[string]*FuncValue{}}; 
}

func (refsh *Refsh) Call(fname string, argv []string){
  function := refsh.Funcs[fname];
  //functype := function.Type().(*FuncType);
  args := refsh.ParseArgs(fname, argv);
  function.Call(args);
}

func CheckArgCount(fname string, argv []string, nargs int){
  if nargs != len(argv){
    fmt.Fprintln(os.Stderr, "Error calling", fname, argv, ": needs", nargs, "arguments.");
    os.Exit(-1);
  }
}

func ParseArg(arg string, argtype Type) Value{
  switch argtype.Name(){
    default: 
      fmt.Fprintln(os.Stderr, "Do not know how to parse", argtype);
      os.Exit(-2);
    case "int":
      return NewValue(ParseInt(arg));
  }
  return NewValue(666); // is never reached.
}

func ParseInt(str string) int{
  i, err := strconv.Atoi(str);
  if err != nil{
      fmt.Fprintln(os.Stderr, "Could not parse to int:", str);
      os.Exit(-3);
  }
  return i;
}

func (refsh *Refsh) ParseArgs(fname string, argv []string) []Value{
  function := refsh.Funcs[fname];
  functype := function.Type().(*FuncType);
  nargs := functype.NumIn();

  CheckArgCount(fname, argv, nargs);

  args := make([]Value, nargs);
  for i:=range(args){
    args[i] = ParseArg(argv[i], functype.In(i));
  }
  return args;
}


func main(){
  refsh := NewRefsh();
  refsh.Funcs["test"] = NewValue(Test).(*FuncValue);
  refsh.Call("test", []string{"123"});
}


func Test(i int){
  fmt.Println("Hello reflection!", i);
}



