package sim

// TODO obviously...
type Solver interface{}

// stores the common data for fixed-step solvers
type SolverState struct {
	Dt float
	Field
}
