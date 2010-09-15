package sim

import (
	"log"
	"os"
	"fmt"
)

// crashes if the test is false
func assert(test bool) {
	if !test {
		log.Crash("assertion failed")
	}
}


// puts a 3 in front of the array
func Size4D(size3D []int) []int {
	assert(len(size3D) == 3)
	size4D := make([]int, 4)
	size4D[0] = 3
	for i := range size3D {
		size4D[i+1] = size3D[i]
	}
	return size4D
}


// removes the 3 in front of the array
func Size3D(size4D []int) []int {
	assert(len(size4D) == 4)
	assert(size4D[0] == 3)
	size3D := make([]int, 3)
	for i := range size3D {
		size3D[i] = size4D[i+1]
	}
	return size3D
}


var Verbosity int = 2

func Warn(msg ...interface{}) {
	fmt.Fprint(os.Stderr, BOLD+"WARNING: "+RESET)
	fmt.Fprint(os.Stderr, msg)
	fmt.Fprint(os.Stderr, ERASE) // Erase rest of line
	fmt.Fprintln(os.Stderr)
}

func Debug(msg ...interface{}) {
	if Verbosity > 0 {
		fmt.Fprint(os.Stderr, msg)
		fmt.Fprint(os.Stderr, ERASE) // Erase rest of line
		fmt.Fprintln(os.Stderr)
	}
}

func Debugv(msg ...interface{}) {
	if Verbosity > 1 {
		Debug(msg)
	}
}


func Debugvv(msg ...interface{}) {
	if Verbosity > 2 {
		Debug(msg)
	}
}

func Error(msg ...interface{}) {
	fmt.Fprint(os.Stderr, msg)
	eraseln()
	os.Exit(-2)
}
