package main

/**
 * Program for calling config.go
 *
 */

import (
	"tensor"
	"sim"
	"os"
	"fmt"
)

const (
	X = iota
	Y
	Z
)

var units *Units = NewUnits()

func init() {
	units.CellSize = []float{1., 1., 1.}
}

func main() {
	commands, args := parseArgs()
	for i := range commands {
		exec(commands[i], args[i])
	}
	//   for i:=range(units.CellSize){
	//     units.CellSize[i] /= units.UnitLength();
	//   }
	tensor.Write(os.Stdout, m)
}

var m *tensor.Tensor4


func exec(command string, args []string) {
	switch command {
	case "--size":
		units.Size = parseSize(args)
		m = tensor.NewTensor4([]int{3, units.Size[X], units.Size[Y], units.Size[Z]})
	case "--cellsize":
		units.CellSize = parseCellSize(args)
	case "--aexch":
		argCount(command, args, 1, 1)
		units.AExch = Atof(args[0])
	case "--msat":
		argCount(command, args, 1, 1)
		units.MSat = Atof(args[0])
	case "--uniform":
		argCount(command, args, 3, 3)
		sim.Uniform(m, Atof(args[X]), Atof(args[Y]), Atof(args[Z])) // a package sim/config would be nice
	default:
		fmt.Fprintln(os.Stderr, "unknown command:", command)
		os.Exit(-1)
	}

}

func parseSize(args []string) []int {
	argCount("size", args, 3, 3)
	size := make([]int, 3)
	for i := 0; i < 3; i++ {
		size[i] = Atoi(args[i])
	}
	return size
}

func parseCellSize(args []string) []float {
	argCount("size", args, 3, 3)
	size := make([]float, 3)
	for i := 0; i < 3; i++ {
		size[i] = Atof(args[i])
	}
	return size
}
