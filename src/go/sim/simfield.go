//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import (
	"fmt"
	. "math"
)

// This file implements the methods for defining
// the applied magnetic field.


// Apply a static field defined in Tesla
func (s *Sim) StaticField(hx, hy, hz float) {
	s.AppliedField = &staticField{[3]float{hx, hy, hz}} // pass it on in tesla so that it stays independent of other problem parameters
	s.Println("Applied field: static, (", hx, ", ", hy, ", ", hz, ") T")
}

type staticField struct {
	b [3]float
}

func (field *staticField) GetAppliedField(time float64) [3]float {
	return field.b
}


// Apply an alternating field
func (s *Sim) RfField(hx, hy, hz float, freq float64) {
	s.AppliedField = &rfField{[3]float{hx, hy, hz}, freq}
	s.Println("Applied field: RF, (", hx, ", ", hy, ", ", hz, ") T, frequency: ", freq, " Hz")
}

type rfField struct {
	b    [3]float
	freq float64
}

func (field *rfField) GetAppliedField(time float64) [3]float {
	sin := float(Sin(field.freq * Pi * time))
	return [3]float{field.b[X] * sin, field.b[Y] * sin, field.b[Z] * sin}
}


// Apply a rotating field
func (s *Sim) RotatingField(hx, hy, hz float, freq float64, phaseX, phaseY, phaseZ float64) {
	s.AppliedField = &rotatingField{[3]float{hx, hy, hz}, freq, [3]float64{phaseX, phaseY, phaseZ}}
	s.Println("Applied field: Rotating, (", hx, ", ", hy, ", ", hz, ") T, frequency: ", freq, " Hz", " phases: ", phaseX, ", ", phaseY, ", ", phaseZ, " rad")
}

type rotatingField struct {
	b     [3]float
	freq  float64
	phase [3]float64
}

func (field *rotatingField) GetAppliedField(time float64) [3]float {
	sinX := float(Sin(field.freq*Pi*time + field.phase[X]))
	sinY := float(Sin(field.freq*Pi*time + field.phase[Y]))
	sinZ := float(Sin(field.freq*Pi*time + field.phase[Z]))
	return [3]float{field.b[X] * sinX, field.b[Y] * sinY, field.b[Z] * sinZ}
}


// Apply a rotating burst
func (s *Sim) RotatingBurst(h float, freq, phase, risetime, duration float64) {
	s.AppliedField = &rotatingBurst{h, freq, phase, risetime, duration}
	s.Println("Applied field: Rotating burst, ", h, " T, frequency: ", freq, " Hz ", "phase between X-Y: ", phase, " risetime: ", risetime, " s", ", duration: ", duration, " s")
}

type rotatingBurst struct {
	b                  float
	freq, phase        float64
	risetime, duration float64
}

func (field *rotatingBurst) GetAppliedField(time float64) [3]float {
	sinx := float(Sin(field.freq * Pi * time))
	siny := float(Sin(field.freq*Pi*time + field.phase))
	norm := float(0.25 * (Erf(time/(field.risetime/2.)-2) + 1) * (2 - Erf((time-field.duration)/(field.risetime/2.)) - 1))
	b := field.b
	return [3]float{0, b * norm * sinx, b * norm * siny}
}


// Control the accuracy of the demag kernel.
// 2^accuracy points are used to integrate the field.
// A high value is accurate and slows down (only) the initialization.
func (s *Sim) DemagAccuracy(accuracy int) {
	s.Println("Demag accuracy:", accuracy)
	if accuracy < 4 {
		s.Warn("Low demag accuracy: " + fmt.Sprint(accuracy))
	}
	s.input.demag_accuracy = accuracy
	s.invalidate()
}


// Calculates the effective field of m and stores it in h
func (s *Sim) calcHeff(m, h *DevTensor) {
	// (1) Self-magnetostatic field
	// The convolution may include the exchange field
	s.Convolve(m, h)

	// (2) Add the externally applied field
	if s.AppliedField != nil {
		s.hext = s.GetAppliedField(s.time * float64(s.UnitTime()))
		for i := range s.hComp {
			s.AddConstant(s.hComp[i], s.hext[i]/s.UnitField())
		}
	}
}
