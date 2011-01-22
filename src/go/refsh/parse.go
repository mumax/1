//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package refsh


// This file implements parsing of function arguments to values


import (
	. "reflect"
	"fmt"
	"os"
	"strconv"
	"strings"
	"math"
)


// INTERNAL
// Parses the argument list "argv" to values suited for the function named by "fname".
func (refsh *Refsh) parseArgs(fname string, argv []string) []Value {
	function := refsh.resolve(fname)
	nargs := function.NumIn()

	if nargs != len(argv) {
		fmt.Fprintln(os.Stderr, "Error calling", fname, argv, ": needs", nargs, "arguments.")
		os.Exit(-1)
	}

	args := make([]Value, nargs)
	for i := range args {
		args[i] = parseArg(argv[i], function.In(i))
	}
	return args
}


// INTERNAL
// Parses a string representation of a given type to a value
// TODO: we need to return Value, err
func parseArg(arg string, argtype Type) Value {
	switch argtype.Name() {
	default:
		panic(fmt.Sprint("Do not know how to parse ", argtype))
	case "int":
		return NewValue(parseInt(arg))
	case "int64":
		return NewValue(parseInt64(arg))
	case "float":
		return NewValue(parseFloat(arg))
	case "float32":
		return NewValue(parseFloat32(arg))
	case "float64":
		return NewValue(parseFloat64(arg))
	case "string":
		return NewValue(arg)
	}
	panic("Bug") // is never reached.
	return NewValue(666)
}

// INTERNAL
func parseInt(str string) int {
	i, err := strconv.Atoi(str)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Could not parse to int:", str)
		os.Exit(-3)
	}
	return i
}

// INTERNAL
func parseInt64(str string) int64 {
	i, err := strconv.Atoi64(str)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Could not parse to int64:", str)
		os.Exit(-3)
	}
	return i
}

// INTERNAL
func parseFloat(str string) float {
	if str == "inf" {
		return float(math.Inf(1))
	}
	if str == "-inf" {
		return float(math.Inf(-1))
	}
	i, err := strconv.Atof(strings.ToLower(str))
	if err != nil {
		fmt.Fprintln(os.Stderr, "Could not parse to float:", str)
		os.Exit(-3)
	}
	return i
}

// INTERNAL
func parseFloat64(str string) float64 {
	if str == "inf" {
		return math.Inf(1)
	}
	if str == "-inf" {
		return math.Inf(-1)
	}
	i, err := strconv.Atof64(strings.ToLower(str))
	if err != nil {
		fmt.Fprintln(os.Stderr, "Could not parse to float64:", str)
		os.Exit(-3)
	}
	return i
}

// INTERNAL
func parseFloat32(str string) float32 {
	if str == "inf" {
		return float32(math.Inf(1))
	}
	if str == "-inf" {
		return float32(math.Inf(-1))
	}
	i, err := strconv.Atof32(strings.ToLower(str))
	if err != nil {
		fmt.Fprintln(os.Stderr, "Could not parse to float32:", str)
		os.Exit(-3)
	}
	return i
}
