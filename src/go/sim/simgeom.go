package sim

func (s *Sim) Size(x, y, z int) {
	s.size[X] = x
	s.size[Y] = y
	s.size[Z] = z
	s.invalidate()
}

func (s *Sim) CellSize(x, y, z float) {
	s.cellsize[X] = x
	s.cellsize[Y] = y
	s.cellsize[Z] = z
	s.invalidate()
}
