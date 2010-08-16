package sim


// Stores a simulation state
// Here, all parameters are STILL IN SI UNITS
// when Sim.init() is called, a solver is initiated with these values converted to internal units.
// We need to keep the originial SI values in case a parameter gets changed during the simulation and we need to re-initialize everything.
type Sim struct {
	// material parameters
	aexch float
	msat  float
	alpha float

	// geometry
	size     [3]int
	cellsize [3]float

	// time stepping
	dt   float
	time float

	// backend
	backend int

	solver *Euler
}

func New() *Sim {
	return NewSim()
}

func NewSim() *Sim {
	return new(Sim)
}

// when a parmeter is changed, the simulation state is invalid until it gets (re-)initialized by init()
func (s *Sim) invalidate() {
	s.solver = nil
}

// when it returns false, init() needs to be called before running
func (s *Sim) isValid() bool {
	return s.solver != nil
}

// (re-)initialize the simulation tree, necessary before running
func (s *Sim) init() {
	if s.isValid() {
		return //no work to do
	}

	dev := CPU

	mat := NewMaterial()
	mat.MSat = s.msat
	mat.AExch = s.aexch
	mat.Alpha = s.alpha

	size := s.size[0:]
	L := mat.UnitLength()
	cellsize := []float{s.cellsize[X] / L, s.cellsize[Y] / L, s.cellsize[Z] / L}

	magnet := NewMagnet(dev, mat, size, cellsize)

	dt := s.dt / mat.UnitTime()
	solver := NewEuler(dev, magnet, dt)

	fmt.Println(solver)

	m := tensor.NewTensorN(Size4D(magnet.Size()))
	for i := range m.List() {
		m.List()[i] = 1.
	}
	TensorCopyTo(m, solver.M())
/*
	file := 0
	for i := 0; i < 100; i++ {
		TensorCopyFrom(solver.M(), m)
		fname := "m" + fmt.Sprintf("%06d", file) + ".t"
		file++
		tensor.WriteFile(fname, m)
		for j := 0; j < 100; j++ {
			solver.Step()
		}
	}

	solver.Dt = 0.01E-12 / mat.UnitTime()
	solver.Alpha = 0.02
	B := solver.UnitField()
	solver.Hext = []float{0 / B, 4.3E-3 / B, -24.6E-3 / B}*/

}

func (s *Sim) AExch(a float) {
	s.aexch = a
	s.invalidate()
}

func (s *Sim) MSat(ms float) {
	s.msat = ms
	s.invalidate()
}

func (s *Sim) Alpha(a float) {
	s.alpha = a
	s.invalidate()
}

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

func (s *Sim) Dt(t float) {
	s.dt = t
	s.invalidate()
}

func (s *Sim) Run(time float) {
	init()
	stop := s.time + time
	for s.time < stop {
		s.solver.Step()
		s.time += s.dt
	}
}


func (s *Sim) Backend(b string) {
	panic("todo")
}