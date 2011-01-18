//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

// Utilities for executing external commands

import (
	"os"
	"exec"
)

// Wrapper for exec.Run.
// Uses the current working directory and environment.
func subprocess(command string, args []string, stdin, stdout, stderr int) (cmd *exec.Cmd, err os.Error) {
	allargs := []string{command} // argument 1, not argument 0 is the first real argument, argument 0 is the program name
	allargs = append(allargs, args...)

	wd, errwd := os.Getwd()
	if errwd != nil {
		err = errwd
		return
	}

	cmd, err = exec.Run(command, allargs, os.Environ(), wd, stdin, stdout, stderr)
	return
}


// Runs the subprocess and waits for it to finish.
// The command is looked up in the PATH.
// Output is passed through to stdout/stderr.
// Typically used for simple system commands: rm, mkfifo, cp, ... 
func syscommand(command string, args []string) (err os.Error){
	command, err = exec.LookPath(command)
	if err != nil{return}
	cmd, err2 := subprocess(command, args, exec.DevNull, exec.PassThrough, exec.PassThrough)
	err = err2
	if err != nil{return}
	_, err = cmd.Wait(0)
	return
}
