//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import (
	"tensor"
)


/**
 * 6-Neighbor exchange kernel
 *
 * Note on self-contributions and the energy density:
 *
 * Contributions to H_eff that are parallel to m do not matter.
 * They do not influnce the dynamics and only add a constant term to the energy.
 * Therefore, the self-contribution of the exchange field can be neglected. This
 * term is -N*m for a cell in a cubic grid, with N the number of neighbors.
 * By neglecting this term, we do not need to take into account boundary conditions.
 * Because the interaction can then be written as a convolution, we can simply
 * include it in the demag convolution kernel and we do not need a separate calculation
 * of the exchange field anymore: an elegant and efficient solution.
 * The dynamics are still correct, only the total energy is offset with a constant
 * term compared to the usual - M . H. Outputting H_eff becomes less useful however,
 * it's better to look at torques. Away from the boundaries, H_eff is "correct".
 */
func Exch6NgbrKernel(size []int, cellsize []float) []*tensor.Tensor3 {
	k := make([]*tensor.Tensor3, 6)
	for i := range k {
		k[i] = tensor.NewTensor3(size)
	}

	for s := 0; s < 3; s++ { // source index Ksdxyz
		i := KernIdx[s][s]
		k[i].Array()[0][0][0] = -2./(cellsize[X]*cellsize[X]) - 2./(cellsize[Y]*cellsize[Y]) - 2./(cellsize[Z]*cellsize[Z])

		for dir := X; dir <= Z; dir++ {
			for side := -1; side <= 1; side += 2 {
				index := make([]int, 3)
				i = KernIdx[s][s]
				index[dir] = wrap(side, size[dir])
				k[i].Array()[index[X]][index[Y]][index[Z]] = 1. / (cellsize[dir] * cellsize[dir])
			}
		}
	}
	return k
}
