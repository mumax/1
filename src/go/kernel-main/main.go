//
package main

import (
	"tensor"
	"sim"
	"os"
	"fmt"
)


func main() {
	commands, args := parseArgs()
	for i := range commands {
		exec(commands[i], args[i])
	}
	for i := range units.CellSize {
		units.CellSize[i] /= units.UnitLength()
	}
	//units.PrintInfo(os.Stderr);
	makeKernel()
}

func makeKernel() {
	demag := sim.FaceKernel(units.Size, units.CellSize, 8)
	exch := sim.Exch6NgbrKernel(units.Size, units.CellSize)
	kernel := tensor.Add(exch, demag)
	tensor.Write(os.Stdout, kernel)
}

var units Units = *NewUnits()

var demagtype string = "cuboid"
var exchtype string = "exch6"


func exec(command string, args []string) {
	switch command {
	case "--size":
		units.Size = parseSize(args)
	case "--cellsize":
		units.CellSize = parseCellSize(args)
	case "--aexch":
		argCount(command, args, 1, 1)
		units.AExch = Atof(args[0])
	case "--msat":
		argCount(command, args, 1, 1)
		units.MSat = Atof(args[0])
	case "--dipole":
		demagtype = "dipole"
	case "--cuboid":
		demagtype = "cuboid"
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
