package sim

// time-dependent applied field

type AppliedField interface {
	// Return the field in SI, time in SI
	GetAppliedField(time float64) [3]float
}
