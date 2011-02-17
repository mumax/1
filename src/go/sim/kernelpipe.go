//  This file is part of MuMax, a high-performance micromagnetic simulator
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

// This file implements piping a kernel from a subprocess.

import(
	. "mumax/common"
	"mumax/tensor"
	"exec"
)

// Executes a subprocess that calculates a micromagnetic kernel.
// The subprocess must write 6 rank-3 tensors to stdout:
// Kxx, Kyy, Kzz, Kyz, Kxz, Kxy.
func PipeKernel(command string, args []string) []*tensor.T3{
	k := make([]*tensor.T3, 6)

	cmd, err := subprocess(command, args, exec.DevNull, exec.Pipe, exec.PassThrough)
	CheckErr(err, ERR_SUBPROCESS)

	for i := range k{
		k[i] = tensor.ToT3(tensor.Read(cmd.Stdout))
	}
	
	return k
}

