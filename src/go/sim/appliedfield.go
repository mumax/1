package sim

// time-dependent applied field

type AppliedField interface {
	// Field at the given time, SI units
	Eval(time float64) [3]float
}
