//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import(
	"exec"
	"iotool"
	"fmt"
	"os"
)

// Main for python ".py" input files
func main_python(infile string) {

	outdir := RemoveExtension(infile) + ".out"
	os.Mkdir(outdir, 0777)//TODO: permission?

	fname_to_py := outdir + "/" + ".fifo-to-python.py"
	syscommand("mkfifo", []string{fname_to_py})

	py_args := []string{fname_to_py}
	py_bin, errlook := exec.LookPath("python")
	Check(errlook, ERR_SUBPROCESS)
	fmt.Println("starting ", py_bin)
	python, errpy := subprocess(py_bin, py_args, exec.DevNull, exec.Pipe, exec.Pipe)
	Check(errpy, ERR_SUBPROCESS)
	fmt.Println("python PID: ", python.Pid)
	go Pipe(python.Stdout, os.Stdout) // TODO: logging etc
	go Pipe(python.Stderr, os.Stderr)

	fmt.Println("Opening FIFO: ", fname_to_py)
    fifo_to_py := iotool.MustOpenWRONLY(fname_to_py)
	fmt.Println("Opened FIFO: ", fname_to_py)

	fmt.Fprintln(fifo_to_py, "kakaprint(1+1)\n")
	fifo_to_py.Close()

	_, errwait := python.Wait(0) // Wait for exit
	Check(errwait, ERR_SUBPROCESS)
}
