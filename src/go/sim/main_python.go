//  This file is part of MuMax, a high-performance micromagnetic simulator
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import (
	. "mumax/common"
	"exec"
	"fmt"
	"os"
	"syscall"
)

// Main for python ".py" input files
func main_python(infile string) {

	// Create output dir
	outdir := RemoveExtension(infile) + ".out"
	os.Mkdir(outdir, 0777) //TODO: permission?

	// Start python subprocess
	py_bin, errlook := exec.LookPath("python")

	// python is started with wd infile.out, so the py file is located in ../infle.py
	//parent := ParentDir(infile)
	file := Filename(infile)
	py_args := []string{"../" + file}

	py_args = append([]string{py_bin}, py_args...)
	CheckErr(errlook, ERR_SUBPROCESS)
	Println("starting ", py_bin, py_args)
	fmt.Fprintln(os.Stderr, "exec ", py_args)
	py_wd := ReplaceExt(infile, ".out")
	fmt.Fprintln(os.Stderr, "py_wd", py_wd)
	python, errpy := exec.Run(py_bin, py_args, os.Environ(), py_wd, exec.Pipe, exec.Pipe, exec.PassThrough)


	CheckErr(errpy, ERR_SUBPROCESS)
	Println("python PID: ", python.Process.Pid)

	// Start mumax --slave subprocess
	mu_args := passthrough_cli_args()
	mu_args = append(mu_args, "--slave", "--stdin", infile)
	Println("starting ", "mumax ", mu_args)
	mumax, errmu := subprocess(os.Getenv(SIMROOT)+"/"+SIMCOMMAND, mu_args, exec.Pipe, exec.Pipe, exec.PassThrough)
	CheckErr(errmu, ERR_SUBPROCESS)
	Println("mumax slave PID ", mumax.Process.Pid)

	// 2-way communication between python and mumax
	// Python's stdout -> mumax's  stdin
	// Mumax's  stdout -> python's stdin
	go Pipe(python.Stdout, mumax.Stdin)
	go Pipe(mumax.Stdout, python.Stdin)

	// Wait for python and mumax to finish.
	// Wait asynchronously so "... has finished" output
	// has a good chance to come in the correct order.
	// If we were to wait for python first and then for mumax,
	// but python hangs and mumax crashes then we would keep
	// on waiting for python without noting the mumax crash (e.g.).
	wait := make(chan int)
	go func() {
		py_wait, errwait := python.Wait(0) // Wait for exit
		CheckErr(errwait, ERR_SUBPROCESS)
		status := py_wait.ExitStatus()
		Println("python exited with status ", status)
		if status != 0 {
			// kill the sibling
			syscall.Kill(mumax.Process.Pid, 9)
			os.Exit(ERR_SUBPROCESS)
		}
		wait <- 1
	}()

	go func() {
		mu_wait, errwait := mumax.Wait(0)
		CheckErr(errwait, ERR_SUBPROCESS)
		status := mu_wait.ExitStatus()
		Println("mumax child process exited with status ", status)
		if status != 0 {
			// kill the sibling
			syscall.Kill(python.Process.Pid, 9)
			os.Exit(ERR_SUBPROCESS)
		}
		wait <- 1
	}()

	<-wait
	<-wait
	Println("main_python done.")
}
