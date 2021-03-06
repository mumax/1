//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

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
	"os"
	"io"
	"strings"
	"unicode"
	"fmt"
)


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
	r.funcnames = append(r.funcnames, funcname)
	r.funcs = append(r.funcs, (*FuncWrapper)(f.(*FuncValue)))
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
	r.funcnames = append(r.funcnames, funcname)
	r.funcs = append(r.funcs, &MethodWrapper{NewValue(reciever), f})
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
	refsh.Print(prompt)
	line, eof := ReadNonemptyLine(in)
	for !eof {
		cmd := line[0]
		args := line[1:]
		refsh.Call(cmd, args)
		refsh.Print(prompt)
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
		//refsh.Errorln( commands[i], args[i]);
		refsh.Call(commands[i], args[i])
	}
}


// Calls a function. Function name and arguments are passed as strings.
// The function name should first have been added by refsh.Add();
func (refsh *Refsh) Call(fname string, argv []string) []interface{} {
	// Debug
	refsh.Errorln(">>> ", fname, "\t ", argv)
	refsh.CallCount++

	function := refsh.resolve(fname)
	if function == nil {
		refsh.Errorln("Unknown command: \""+fname+"\". Options are:", refsh.funcnames)
		if refsh.CrashOnError {
			os.Exit(-5)
		}
	} else {
		args := refsh.parseArgs(fname, argv)
		retval := function.Call(args)
		ret := make([]interface{}, len(retval))
		for i := range retval {
			ret[i] = retval[i].Interface()
		}
		return ret
	}
	panic("bug")
	return nil
}

type Refsh struct {
	funcnames    []string          // known function or method names (we do not use a map to not exclude the possibility of overloading)
	funcs        []Caller          // functions/methods corresponding to funcnames
	help         map[string]string // help strings corresponding to funcnames
	CrashOnError bool              // crash the program on a syntax error or just report it (e.g. for interactive mode)
	CallCount    int               //counts number of commands executed
	Output       Printer           //Used to print output, may be nil
}


type Printer interface {
	Print(msg ...interface{})
	Println(msg ...interface{})
	Errorln(msg ...interface{})
}


func (refsh *Refsh) Print(msg ...interface{}) {
	if refsh.Output != nil {
		refsh.Output.Print(msg...)
	} else {
		fmt.Print(msg...)
	}
}


func (refsh *Refsh) Println(msg ...interface{}) {
	if refsh.Output != nil {
		refsh.Output.Println(msg...)
	} else {
		fmt.Println(msg...)
	}
}


func (refsh *Refsh) Errorln(msg ...interface{}) {
	if refsh.Output != nil {
		refsh.Output.Errorln(msg...)
	} else {
		fmt.Fprintln(os.Stderr, msg...)
	}
}


func NewRefsh() *Refsh {
	refsh := new(Refsh)
	CAPACITY := 10 // Initial function name capacity, but can grow
	refsh.funcnames = make([]string, CAPACITY)[0:0]
	refsh.funcs = make([]Caller, CAPACITY)[0:0]
	refsh.CrashOnError = true
	// built-in functions
	refsh.AddMethod("include", refsh, "Include")
	return refsh
}


// executes the file
func (refsh *Refsh) Include(file string) {
	in, err := os.Open(file, os.O_RDONLY, 0666)
	if err != nil {
		panic(err)
	}
	refsh.Exec(in)
}
