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
 * it's better to look at torques. Away from the boundaries, H_eff is correct.
 */
func Exch6NgbrKernel(paddedsize []int, cellsize []float) *tensor.Tensor5 {
	size := paddedsize
	k := tensor.NewTensor5([]int{3, 3, size[0], size[1], size[2]})

	for s := 0; s < 3; s++ { // source index Ksdxyz
		k.Array()[s][s][0][0][0] = -2./(cellsize[X]*cellsize[X]) - 2./(cellsize[Y]*cellsize[Y]) - 2./(cellsize[Z]*cellsize[Z])

		for dir := X; dir <= Z; dir++ {
			for side := -1; side <= 1; side += 2 {
				index := make([]int, 5)
				index[0] = s
				index[1] = s
				index[dir+2] = wrap(side, size[dir])
				tensor.Set(k, index, 1./(cellsize[dir]*cellsize[dir]))
			}
		}

	}
	return k
}
