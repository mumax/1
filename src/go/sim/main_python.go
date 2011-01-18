//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import (
	"exec"
	//	"iotool"
	"fmt"
	"os"
)

// Main for python ".py" input files
func main_python(infile string) {

	// Create output dir
	outdir := RemoveExtension(infile) + ".out"
	os.Mkdir(outdir, 0777) //TODO: permission?

	// Start python subprocess
	py_args := []string{infile}
	py_bin, errlook := exec.LookPath("python")
	Check(errlook, ERR_SUBPROCESS)
	fmt.Println("starting ", py_bin)
	python, errpy := subprocess(py_bin, py_args, exec.Pipe, exec.Pipe, exec.PassThrough)
	Check(errpy, ERR_SUBPROCESS)
	fmt.Println("python PID: ", python.Pid)

	// Start mumax --slave subprocess
	mu_args := passthrough_cli_args()
	mu_args = append(mu_args, "--slave", "--stdin", RemoveExtension(infile))
	mumax, errmu := subprocess(os.Getenv(SIMROOT)+"/"+SIMCOMMAND, mu_args, exec.Pipe, exec.Pipe, exec.PassThrough)
	Check(errmu, ERR_SUBPROCESS)
	fmt.Println("mumax slave PID ", mumax.Pid)

	// Plumbing
	go Pipe(python.Stdout, mumax.Stdin)
	go Pipe(mumax.Stdout, python.Stdin)

	pywait := make(chan int)
	go func() {
		_, errwait := python.Wait(0) // Wait for exit
		Check(errwait, ERR_SUBPROCESS)
		fmt.Println("python exited")
		pywait <- 1
	}()
	_, errwait := mumax.Wait(0)
	Check(errwait, ERR_SUBPROCESS)
	fmt.Println("mumax exited")

	<-pywait
}
