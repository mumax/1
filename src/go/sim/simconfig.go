package sim

import "tensor"

func (s *Sim) Uniform(mx, my, mz float) {
	s.ensure_m()
	Uniform(s.m, mx, my, mz)
}

func (s *Sim) ensure_m() {
	if s.m == nil {
		s.m = tensor.NewTensor4([]int{3, s.size[X], s.size[Y], s.size[Z]})
	}
}
