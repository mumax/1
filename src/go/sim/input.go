package sim

// Sim has an "input" member of type Input.
//
// In this struct, all parameters are STILL IN SI UNITS.
// When Sim.init() is called, a solver is initiated
// with these values converted to internal units.
// We need to keep the originial SI values in case a
// parameter gets changed during the simulation and
// we need to re-initialize everything.
//
// This struct is not embedded in Sim but appears as
// a member "input" so that we have to write, e.g.,
// sim.input.dt to make clear it is not necessarily the
// same as sim.dt (which is in internal units)
//
type Input struct {
	aexch          float
	msat           float
	alpha          float
	size           [3]int
	cellsize       [3]float
	demag_accuracy int
	dt             float
	solvertype     string
}
