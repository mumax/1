// refsh is a "reflective shell", an interpreter that parses
// commands, executing them via run-time reflection.
// Usage: first set up a new interpreter:
// sh := refsh.New()
// sh.AddFunc("shortname", Func)
// sh.AddMethod("shortname", &Reciever{...}, "MethodName")
// Then execute some commands:
// sh.Exec(reader)
//
package refsh

import (
	. "reflect"
	"fmt"
	"os"
	"io"
	"strings"
	"unicode"
)

// Maximum number of functions.
// TODO use a vector to make this unlimited.
const CAPACITY = 100

// Makes a new Refsh
func New() *Refsh {
	return NewRefsh()
}

// Adds a function to the list of known commands.
// example: refsh.Add("exit", Exit)
func (r *Refsh) AddFunc(funcname string, function interface{}) {
	f := NewValue(function)

	if r.resolve(funcname) != nil {
		panic("Aldready defined: " + funcname)
	}

	r.funcnames = r.funcnames[0 : len(r.funcnames)+1]
	r.funcnames[len(r.funcnames)-1] = funcname
	r.funcs = r.funcs[0 : len(r.funcs)+1]
	r.funcs[len(r.funcs)-1] = (*FuncWrapper)(f.(*FuncValue))
}

// Adds a method to the list of known commands
// example: refsh.Add("field", reciever, "GetField")
// (command field ... will call reciever.GetField(...))
func (r *Refsh) AddMethod(funcname string, reciever interface{}, methodname string) {
	if r.resolve(funcname) != nil {
		panic("Aldready defined: " + funcname)
	}

	typ := Typeof(reciever)
	var f *FuncValue
	for i := 0; i < typ.NumMethod(); i++ {
		if typ.Method(i).Name == methodname {
			f = typ.Method(i).Func
		}
	}
	if f == nil {
		panic("Method does not exist: " + methodname)
	}

	r.funcnames = r.funcnames[0 : len(r.funcnames)+1]
	r.funcnames[len(r.funcnames)-1] = funcname
	r.funcs = r.funcs[0 : len(r.funcs)+1]

	r.funcs[len(r.funcs)-1] = &MethodWrapper{NewValue(reciever), f}
}

// Adds all the public Methods of the reciever,
// giving them a lower-case command name
func (r *Refsh) AddAllMethods(reciever interface{}) {
	typ := Typeof(reciever)
	for i := 0; i < typ.NumMethod(); i++ {
		name := typ.Method(i).Name
		if unicode.IsUpper(int(name[0])) {
			r.AddMethod(strings.ToLower(name), reciever, name)
		}
	}
}

// parses and executes the commands read from in
// bash-like syntax:
// command arg1 arg2
// command arg1
// #comment
func (refsh *Refsh) Exec(in io.Reader) {
	for line, eof := ReadNonemptyLine(in); !eof; line, eof = ReadNonemptyLine(in) {
		cmd := line[0]
		args := line[1:]
		refsh.Call(cmd, args)
	}
}

const prompt = ">> "

// starts an interactive command line
// When an error is encountered, the program will not abort
// but print a message and continue
// TODO: exit should stop this refsh, not exit the entire program
func (refsh *Refsh) Interactive() {
	in := os.Stdin
	fmt.Print(prompt)
	line, eof := ReadNonemptyLine(in)
	for !eof {
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
func (refsh *Refsh) Call(fname string, argv []string) []Value {
	function := refsh.resolve(fname)
	if function == nil {
		fmt.Fprintln(os.Stderr, "Unknown command: \""+fname+"\". Options are:", refsh.funcnames)
		if refsh.CrashOnError {
			os.Exit(-5)
		}
	} else {
		args := refsh.parseArgs(fname, argv)
		return function.Call(args)
	}
	panic("bug")
	return nil
}

type Refsh struct {
	funcnames    []string
	funcs        []Caller
	CrashOnError bool
}

func NewRefsh() *Refsh {
	refsh := new(Refsh)
	refsh.funcnames = make([]string, CAPACITY)[0:0]
	refsh.funcs = make([]Caller, CAPACITY)[0:0]
	refsh.CrashOnError = true
	return refsh
}
