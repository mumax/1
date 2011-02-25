//  This file is part of MuMax, a high-performance micromagnetic simulator
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import (
	. "mumax/common"
	"container/vector"
	"fmt"
	"math"
	"strings"
)

// This file implements the methods for defining
// the applied magnetic field.
// TODO: we need a time 0 !


//                                                                      IMPORTANT: this is one of the places where X,Y,Z get swapped
//                                                                      what is (X,Y,Z) internally becomes (Z,Y,X) for the user!


// Choose the kernel type (command for subprogram, e.g., mumaxkern-go). Mainly intended for debugging.
func (s *Sim) KernelType(command string) {
	s.input.kernelType = command
	s.invalidate()
}

// Override if exchange should be included in convolution. Mainly intended for debugging.
func (s *Sim) ExchInConv(exchInConv bool) {
	s.exchInConv = exchInConv
	s.invalidate()
}

// Set the exchange type
func (s *Sim) ExchType(exchType int) {
	s.input.exchType = exchType
	s.invalidate()
}


func (s *Sim) apply(what string, function AppliedField){
	switch strings.ToLower(what){
		default: panic(InputErr("Illegal Argument: " + what + " Options are: [field, current]"))
		case "field": s.appliedField = function
		case "current": s.appliedCurrent = function
	}
}

// Apply a static field defined in Tesla
func (s *Sim) ApplyStatic(what string, hz, hy, hx float32) {
	s.apply(what, &staticField{[3]float32{hx, hy, hz}} )
	s.Println("Applied " + what + ": static, (", hz, ", ", hy, ", ", hx, ") T")
}

type staticField struct {
	b [3]float32
}

func (field *staticField) GetAppliedField(time float64) [3]float32 {
	return field.b
}


// Apply a field defined by a number of points
func (s *Sim) UsePointwiseField(){

}

type pointwiseField struct{
	points vector.Vector
}

func (s *Sim) ApplyPulse(what string, hz, hy, hx float32, duration, risetime float64) {
	s.apply(what, &pulsedField{[3]float32{hx, hy, hz}, duration, risetime})
	s.Println("Applied " + what + ": pulse, (", hx, ", ", hy, ", ", hz, ") T, ", duration, "s FWHM, ", risetime, "s rise- and falltime (0-100%)")
}

type pulsedField struct {
	b                  [3]float32
	duration, risetime float64
}


func (f *pulsedField) GetAppliedField(time float64) [3]float32 {
	var scale float64

	if time > 0 && time < f.risetime {
		scale = 0.5 - 0.5*math.Cos(time*math.Pi/f.risetime)
	} else if time >= f.risetime && time < f.duration+f.risetime {
		scale = 1.
	} else if time >= f.duration+f.risetime && time < f.duration+2.*f.risetime {
		scale = 0.5 + 0.5*math.Cos((time-f.duration-f.risetime)*math.Pi/f.risetime)
	}
	scale32 := float32(scale)
	return [3]float32{scale32 * f.b[0], scale32 * f.b[1], scale32 * f.b[2]}
}


// Apply an alternating field
func (s *Sim) ApplyRf(what string, hz, hy, hx float32, freq float64) {
	s.apply(what, &rfField{[3]float32{hx, hy, hz}, freq})
	s.Println("Applied field: RF, (", hx, ", ", hy, ", ", hz, ") T, frequency: ", freq, " Hz")
}

type rfField struct {
	b    [3]float32
	freq float64
}

func (field *rfField) GetAppliedField(time float64) [3]float32 {
	sin := float32(math.Sin(field.freq * 2 * math.Pi * time))
	return [3]float32{field.b[X] * sin, field.b[Y] * sin, field.b[Z] * sin}
}


func (s *Sim) ApplySawtooth(what string, hz, hy, hx float32, freq float64) {
	var st sawtooth
	st = sawtooth(rfField{[3]float32{hx, hy, hz}, freq})
	s.apply(what, &st)
	s.Println("Applied field: sawtooth, (", hx, ", ", hy, ", ", hz, ") T, frequency: ", freq, " Hz")
}

type sawtooth rfField

func (field *sawtooth) GetAppliedField(time float64) [3]float32 {
	sin := float32(math.Asin(math.Sin(field.freq*2*math.Pi*time)) / (math.Pi / 2))
	return [3]float32{field.b[X] * sin, field.b[Y] * sin, field.b[Z] * sin}
}

// Apply a rotating field
func (s *Sim) ApplyRotating(what string, hz, hy, hx float32, freq float64, phaseX, phaseY, phaseZ float64) {
	s.apply(what, &rotatingField{[3]float32{hx, hy, hz}, freq, [3]float64{phaseX, phaseY, phaseZ}})
	s.Println("Applied field: Rotating, (", hx, ", ", hy, ", ", hz, ") T, frequency: ", freq, " Hz", " phases: ", phaseX, ", ", phaseY, ", ", phaseZ, " rad")
}

type rotatingField struct {
	b     [3]float32
	freq  float64
	phase [3]float64
}

func (field *rotatingField) GetAppliedField(time float64) [3]float32 {
	sinX := float32(math.Sin(field.freq*2*math.Pi*time + field.phase[X]))
	sinY := float32(math.Sin(field.freq*2*math.Pi*time + field.phase[Y]))
	sinZ := float32(math.Sin(field.freq*2*math.Pi*time + field.phase[Z]))
	return [3]float32{field.b[X] * sinX, field.b[Y] * sinY, field.b[Z] * sinZ}
}


// Apply a rotating burst.
// phase: -pi/2=CW, pi/2=CCW
func (s *Sim) ApplyRotatingBurst(what string, h float32, freq, phase, risetime, duration float64) {
	s.apply(what, &rotatingBurst{h, freq, phase, risetime, duration})
	s.Println("Applied field: Rotating burst, ", h, " T, frequency: ", freq, " Hz ", "phase between X-Y: ", phase, " risetime: ", risetime, " s", ", duration: ", duration, " s")
}

type rotatingBurst struct {
	b                  float32
	freq, phase        float64
	risetime, duration float64
}

const SQRT2 = 1.414213562373095


func (field *rotatingBurst) GetAppliedField(time float64) [3]float32 {
	sinx := float32(math.Sin(field.freq * 2 * math.Pi * time))
	siny := float32(math.Sin(field.freq*2*math.Pi*time + field.phase))
	norm := float32(0.25 / SQRT2 * (math.Erf(time/(field.risetime/2.)-2) + 1) * (2 - math.Erf((time-field.duration)/(field.risetime/2.)) - 1))
	b := field.b
	return [3]float32{0, b * norm * sinx, b * norm * siny}
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
	// (1) Self-magnetostatic field and exchange
	// The convolution may include the exchange field
	if s.input.wantDemag {
		s.Convolve(m, h)
	} else {
		ZeroTensor(h)
	}
	if !s.input.wantDemag || !s.exchInConv {
		s.AddExch(m, h)
	}

	// (2) Add the externally applied field
	if s.appliedField != nil {
		s.hextSI = s.appliedField.GetAppliedField(s.time * float64(s.UnitTime()))
	} else {
		s.hextSI = [3]float32{0., 0., 0.}
	}

	B := s.UnitField()
	s.hextInt[0] = s.hextSI[0] / B
	s.hextInt[1] = s.hextSI[1] / B
	s.hextInt[2] = s.hextSI[2] / B

	// TODO: only if needed
	s.AddLocalFields(m, h, s.hextInt, s.input.anisType, s.anisKInt, s.input.anisAxes)

	// (3) Add the edge-correction field
	if s.input.edgeCorr != 0 {
		s.addEdgeField(m, h)
	}
}
