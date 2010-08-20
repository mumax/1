package sim

// This file implements methods for generating
// initial magnetization configurations like
// vortices, Landau patterns, etc.

import (
	"tensor"
	"rand"
)

// INTERNAL: to be called before setting a magnetization state,
// ensures local memory for m has been allocated already
func (s *Sim) ensure_m() {
	if s.m == nil {
		s.m = tensor.NewTensor4([]int{3, s.size[X], s.size[Y], s.size[Z]})
	}
}

// Make the magnetization uniform. (mx, my, mz) needs not to be normalized
func (s *Sim) Uniform(mx, my, mz float) {
	s.ensure_m()
	a := s.m.Array()
	for i := range a[0] {
		for j := range a[0][i] {
			for k := range a[0][i][j] {
				a[X][i][j][k] = mx
				a[Y][i][j][k] = my
				a[Z][i][j][k] = mz
			}
		}
	}
}

// Adds noise with the specified amplitude to the magnetization state.
// Handy to break the symmetry.
func (s *Sim) AddNoise(amplitude float) {
	s.ensure_m()
	amplitude *= 2
	list := s.m.List()
	for i := range list {
		list[i] += amplitude * (rand.Float() - 0.5)
	}
}
