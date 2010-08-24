package sim


type Solver interface{
  Step()
}

// stores the common data for fixed-step solvers
type SolverState struct {
	Dt float
	*Field
}

func(s *SolverState) SetDt(dt float){
  s.Dt = dt
}
