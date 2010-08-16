package sim


// Stores a simulation state
// Here, all parameters are STILL IN SI UNITS
// when Sim.init() is called, a solver is initiated with these values converted to internal units.
// We need to keep the originial SI values in case a parameter gets changed during the simulation and we need to re-initialize everything.
type Sim struct {
	// material parameters
	aexch float64
	msat  float64
	alpha float64

	// geometry
	size     [3]int
	cellsize [3]float64

	// time stepping
	dt float64

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

// when a parmeter is changed, the simulation state is invalid until it gets (re-)initialized
func (s *Sim) invalidate() {
	s.solver = nil
}

func (s *Sim) AExch(a float64) {
	s.aexch = a
	s.invalidate()
}

func (s *Sim) MSat(ms float64) {
	s.msat = ms
	s.invalidate()
}

func (s *Sim) Alpha(a float64) {
	s.alpha = a
	s.invalidate()
}

func (s *Sim) Size(x, y, z int) {
	s.size[X] = x
	s.size[Y] = y
	s.size[Z] = z
	s.invalidate()
}

func (s *Sim) CellSize(x, y, z float64) {
	s.cellsize[X] = x
	s.cellsize[Y] = y
	s.cellsize[Z] = z
	s.invalidate()
}

func (s *Sim) Dt(t float64) {
	s.dt = t
	s.invalidate()
}

func (s *Sim) Run(time float64) {

}


func (s *Sim) Backend(b string) {
	panic("todo")
}
