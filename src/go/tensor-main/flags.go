//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package main


import (
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"log"
)


func argCount(command string, args []string, min, max int) {
	if len(args) < min || len(args) > max {
		if min != max {
			fmt.Fprintln(os.Stderr, command, "expects", min, "to", max, "parameters")
		} else {
			fmt.Fprintln(os.Stderr, command, "expects", min, "parameters")
		}
		os.Exit(-1)
	}
}

/**
 * Looks for a pattern like: --command0 arg0 arg1 --command1 arg0 --command2 ...
 * Returns an array of commands and args. command[i] has args args[i].
 */
func parseArgs() ([]string, [][]string) {
	ncommands := 0
	for i := 0; i < flag.NArg(); i++ {
		if strings.HasPrefix(flag.Arg(i), "--") {
			ncommands++
		}
	}

	commands := make([]string, ncommands)
	args := make([][]string, ncommands)

	{
		command := 0
		i := 0
		for i < flag.NArg() {
			//assert(strings.HasPrefix(flag.Arg(i), "--"));
			if !strings.HasPrefix(flag.Arg(i), "--") {
				fmt.Println("Expecting a command starting with --")
				os.Exit(-2)
			}
			commands[command] = flag.Arg(i)
			nargs := 0
			i++
			for i < flag.NArg() && !strings.HasPrefix(flag.Arg(i), "--") {
				nargs++
				i++
			}
			args[command] = make([]string, nargs)
			command++
		}
	}

	{
		command := 0
		i := 0
		for i < flag.NArg() {
			assert(strings.HasPrefix(flag.Arg(i), "--"))
			commands[command] = flag.Arg(i)
			nargs := 0
			i++
			for i < flag.NArg() && !strings.HasPrefix(flag.Arg(i), "--") {
				args[command][nargs] = flag.Arg(i)
				nargs++
				i++
			}
			command++
		}
	}
	return commands, args
}

/**
 * String to int conversion with a few special cases like x->0, y->1, ...
 * Aborts wit error message on illegal input.
 */
func Atoi(s string) int {

	switch strings.ToLower(s) {
	case "x":
		return 0
	case "y":
		return 1
	case "z":
		return 2
	}

	i, err := strconv.Atoi(s)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Expecting integer argument:", err)
		os.Exit(-1)
	}
	return i
}

/**
 * String to float conversion.
 * Aborts wit error message on illegal input.
 */
func Atof(s string) float {
	s = strings.ToLower(s) // otherwise it does not want to parse 1E-9
	f, err := strconv.Atof(s)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Expecting floating-point argument:", err)
		os.Exit(-1)
	}
	return f
}

/**
 * Returns the file with the name given as argument,
 * or stdout when the name is omitted.
 */
func parseFileOrStdout(args []string) *os.File {
	argCount("an output function", args, 0, 1)
	if len(args) == 0 {
		return os.Stdout
	}
	return fOpenz(args[0])
}


/**
 * Returns the file with the name given as argument,
 * or stdin when the name is omitted.
 */
func parseFileOrStdin(args []string) *os.File {
	argCount("an output function", args, 0, 1)
	if len(args) == 0 {
		return os.Stdin
	}
	return fOpenz(args[0])
}


/** Crashes the program when the test is false. */
func assert(test bool) {
	if !test {
		log.Crash("Assertion failed")
	}
}

/** Crashes the program with an error message when the test is false. */
func assertMsg(test bool, msg string) {
	if !test {
		log.Crash(msg)
	}
}

// TODO: rename function, FOpen already exists.
/** Todo: sometimes appends instead of overwriting... */
func fOpenz(filename string) *os.File {
	file, ok := os.Open(filename, os.O_RDWR|os.O_CREAT, 0666)
	if ok != nil {
		fmt.Fprint(os.Stderr, ok, "\n")
		log.Crash("Could not open file")
	}
	return file
}
