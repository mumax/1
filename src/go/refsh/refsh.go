package refsh

import (
	. "reflect"
	"fmt"
	"os"
	"strconv"
	"io"
// 	"scanner"
// 	"container/vector"
	//    "runtime"
)

/**
 * Maximum number of functions.
 * @todo use a vector to make this unlimited.
 */
const CAPACITY = 100


type Refsh struct {
	funcnames    []string
	funcs        []*FuncValue
	CrashOnError bool
}

func NewRefsh() *Refsh {
	refsh := new(Refsh)
	refsh.funcnames = make([]string, CAPACITY)[0:0]
	refsh.funcs = make([]*FuncValue, CAPACITY)[0:0]
	refsh.CrashOnError = true
	return refsh
}

func New() *Refsh {
	return NewRefsh()
}

// Adds a function to the list of known commands.
// example: refsh.Add("exit", Exit)
func (r *Refsh) Add(funcname string, f interface{}) {
	function := NewValue(f)
	if r.resolve(funcname) != nil {
		fmt.Fprintln(os.Stderr, "Aldready defined:", funcname)
		os.Exit(-4)
	}
	r.funcnames = r.funcnames[0 : len(r.funcnames)+1]
	r.funcnames[len(r.funcnames)-1] = funcname
	r.funcs = r.funcs[0 : len(r.funcs)+1]
	r.funcs[len(r.funcs)-1] = function.(*FuncValue)
}

// parses and executes the commands read from in
// bash-like syntax:
// command arg1 arg2
// command arg1
func (refsh *Refsh) Exec(in io.Reader) {
	for line, eof := ReadNonemptyLine(in); !eof; line, eof = ReadNonemptyLine(in) {
        cmd := line[0]
        args := line[1:]
        refsh.Call(cmd, args)
    }
}

const prompt = ">> "

// starts an interactive command line
// TODO: exit should stop this refsh, not exit the entire program
func (refsh *Refsh) Interactive() {
    in := os.Stdin
    fmt.Print(prompt)
    line, eof := ReadNonemptyLine(in)
	for  !eof {
        cmd := line[0]
        args := line[1:]
        refsh.Call(cmd, args)
        fmt.Print(prompt)
        line, eof = ReadNonemptyLine(in) 
    }
}

func exit() {
	os.Exit(0)
}


// Executes the command line arguments. They should have a syntax like:
// --command1 arg1 arg2 --command2 --command3 arg1
func (refsh *Refsh) ExecFlags() {
	commands, args := ParseFlags()
	for i := range commands {
		//fmt.Fprintln(os.Stderr, commands[i], args[i]);
		refsh.Call(commands[i], args[i])
	}
}


// Calls a function. Function name and arguments are passed as strings.
// The function name should first have been added by refsh.Add();
func (refsh *Refsh) Call(fname string, argv []string) {
	function := refsh.resolve(fname)
	if function == nil {
		fmt.Fprintln(os.Stderr, "Unknown command:", fname, "Options are:", refsh.funcnames)
		if refsh.CrashOnError {
			os.Exit(-5)
		}
	} else {
		args := refsh.parseArgs(fname, argv)
		function.Call(args)
	}
}


// // Reads one line.
// // The first token is returned in command, the rest in the args array
// func readLine(s *scanner.Scanner) (command string, args []string) {
// 	line := s.Pos().Line // I found no direct way to detect line ends using a scanner
// 	startline := line    // so I check if the line number changes
// 
// 	token := s.Scan() // the first token is the command
// 	command = s.TokenText()
// 	line = s.Pos().Line
// 	// the other tokens are the arguments
// 	argl := vector.StringVector(make([]string, 0))
// 	for token != scanner.EOF && line == startline {
// 		token = s.Scan()
// 		argl.Push(s.TokenText())
// 		line = s.Pos().Line
// 	}
// 	args = []string(argl)
// 	return
// }


func (r *Refsh) resolve(funcname string) *FuncValue {
	for i := range r.funcnames {
		if r.funcnames[i] == funcname {
			return r.funcs[i]
		}
	}
	return nil // never reached
}


func (refsh *Refsh) parseArgs(fname string, argv []string) []Value {
	function := refsh.resolve(fname)
	functype := function.Type().(*FuncType)
	nargs := functype.NumIn()

	if nargs != len(argv) {
		fmt.Fprintln(os.Stderr, "Error calling", fname, argv, ": needs", nargs, "arguments.")
		os.Exit(-1)
	}

	args := make([]Value, nargs)
	for i := range args {
		args[i] = parseArg(argv[i], functype.In(i))
	}
	return args
}


func parseArg(arg string, argtype Type) Value {
	switch argtype.Name() {
	default:
		fmt.Fprintln(os.Stderr, "Do not know how to parse", argtype)
		os.Exit(-2)
	case "int":
		return NewValue(parseInt(arg))
	}
	return NewValue(666) // is never reached.
}


func parseInt(str string) int {
	i, err := strconv.Atoi(str)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Could not parse to int:", str)
		os.Exit(-3)
	}
	return i
}

/*

func main(){
  refsh := NewRefsh();
  refsh.Add("test", NewValue(SayHello));
  refsh.ExecFlags();
}

func SayHello(i int){
  fmt.Println("Hello reflection!", i);
}*/
