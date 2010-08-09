package sim

import (
	"testing"
)

func TestEuler(t *testing.T) {
	dt := 1E-6

	size := []int{1, 32, 128}
	cellsize := []float{1., 1., 1.}

	mat := NewMaterial()
	mat.MSat = 800E3
	mat.AExch = 1.1E-13
	mat.Alpha = 1.0
	magnet := NewMagnet(backend, mat, size, cellsize)
	//field := NewField(backend, magnet) // to be constructed by solver

	_ = NewEuler(backend, magnet, dt)

}
