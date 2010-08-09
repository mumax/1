package main

import (
	"fmt"
	. "tensor"
	"os"
)


func main() {
	commands, args := parseArgs()
	if len(commands) == 0 {
		commands = []string{"--read"}
		args = make([][]string, 1)
		args[0] = []string{}
	}
	for c := range commands {
		exec(commands[c], args[c])
	}
	if !written {
		Write(os.Stdout, getbuffer())
		written = true
	}
}

/** stores the tensor that is being maipulated. should be accessed through setbuffer(), getbuffer(). */
var tensor_ Tensor


func setbuffer(t Tensor) {
	tensor_ = t
}

/** gets the buffered tensor, or reads from stdin if no tensor was created yet. */
func getbuffer() Tensor {
	if tensor_ == nil {
		setbuffer(Read(os.Stdin))
	}
	return tensor_
}

/** Has SOME output been written already? If not, we will write the tensor buffer to stdout before the program exits. */
var written bool = false

func exec(command string, args []string) {
	switch command {
	case "--read":
		setbuffer(Read(parseFileOrStdin(args)))
	case "--new":
		rank := len(args)
		size := make([]int, rank)
		for i := range size {
			size[i] = Atoi(args[i])
		}
		setbuffer(NewTensorN(size))

	case "--component":
		argCount(command, args, 1, 1)
		setbuffer(Slice(getbuffer(), 0, Atoi(args[0])))
	case "--slice":
		argCount(command, args, 2, 2)
		setbuffer(Slice(getbuffer(), Atoi(args[0]), Atoi(args[1])))
	case "--transpose":
		argCount(command, args, 2, 2)
		setbuffer(Transpose(getbuffer(), Atoi(args[0]), Atoi(args[1])))

	case "--normalize":
		argCount(command, args, 1, 1)
		setbuffer(Normalize(getbuffer(), Atoi(args[0])))
	case "--average":
		argCount(command, args, 1, 1)
		setbuffer(Average(getbuffer(), Atoi(args[0])))

	case "--get":
		argCount(command, args, Rank(getbuffer()), Rank(getbuffer()))
		t := Buffer(getbuffer())
		index := make([]int, Rank(getbuffer()))
		for i := 0; i < len(args); i++ {
			index[i] = Atoi(args[i])
		}
		fmt.Println(t.Get(index))
		written = true

	case "--set":
		argCount(command, args, Rank(getbuffer())+1, Rank(getbuffer())+1)
		t := Buffer(getbuffer())
		index := make([]int, Rank(getbuffer()))
		for i := 0; i < len(args)-1; i++ {
			index[i] = Atoi(args[i])
		}
		Set(t, index, Atof(args[len(args)-1]))
		setbuffer(t)
	case "--setall":
		argCount(command, args, 1, 1)
		t := Buffer(getbuffer())
		SetAll(t, Atof(args[0]))
		setbuffer(t)

	case "--rank":
		fmt.Fprintln(parseFileOrStdout(args), Rank(getbuffer()))
		written = true
	case "--size":
		out := parseFileOrStdout(args)
		for i := range getbuffer().Size() {
			fmt.Fprint(out, getbuffer().Size()[i], " ")
		}
		fmt.Println()
		written = true
	case "--format":
		Format(parseFileOrStdout(args), getbuffer())
		written = true
	case "--write":
		Write(parseFileOrStdout(args), getbuffer())
		written = true
	case "--gnuplot":
		PrintVectors(parseFileOrStdout(args), getbuffer())
		written = true
	default:
		fmt.Fprintln(os.Stderr, "unknown command:", command)
		os.Exit(-1)
	}
}
