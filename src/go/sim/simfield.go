//  This file is part of MuMax, a high-performance micromagnetic simulator
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import (
	. "mumax/common"
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


func (s *Sim) apply(what string, function AppliedField) {
	switch strings.ToLower(what) {
	default:
		panic(InputErr("Illegal Argument: " + what + " Options are: [field (or B), currentdensiry (or j)]"))
	case "field", "b":
		s.appliedField = function
	case "currentdensity", "j":
		s.appliedCurrDens = function
	}
}


func (s *Sim) getApplied(what string) AppliedField {
	switch strings.ToLower(what) {
	default:
		panic(Bug("Illegal Argument: " + what + " Options are: [field (or B), currentdensiry (or j)]"))
	case "field", "b":
		return s.appliedField
	case "currentdensity", "j":
		return s.appliedCurrDens
	}
	panic(Bug(""))
	return nil
}

// Apply a static field defined in Tesla
func (s *Sim) ApplyStatic(what string, hz, hy, hx float32) {
	s.apply(what, &staticField{[3]float32{hx, hy, hz}})
	s.Println("Applied "+what+": static, (", hz, ", ", hy, ", ", hx, ") T")
}

type staticField struct {
	b [3]float32
}

func (field *staticField) GetAppliedField(time float64) [3]float32 {
	return field.b
}


// Apply a field defined by a number of points
func (s *Sim) ApplyPointwise(what string, time float64, bz, by, bx float32) {
	p, ok := (s.getApplied(what)).(*pointwiseField)
	if !ok {
		println("new pointwise applied " + what)
		p = &pointwiseField{0, make([]fieldpoint, 0)}
		s.apply(what, p)
	}
	p.add(time, bx, by, bz)
}

type pointwiseField struct {
	lastIdx int          // Index of last time, for fast lookup of next
	points  []fieldpoint // stores time, bx, by, bz
}

type fieldpoint struct {
	time float64
	b    [3]float32
}

func (f *pointwiseField) add(t float64, bx, by, bz float32) {
	f.points = append(f.points, fieldpoint{t, [3]float32{bx, by, bz}})
}

func (field *pointwiseField) GetAppliedField(time float64) [3]float32 {

	if len(field.points) < 2 {
		panic(InputErr("Pointwise field/current needs at least two points"))
	}
	//find closest times
	for ; field.lastIdx < len(field.points); field.lastIdx++ {
		if field.points[field.lastIdx].time >= time {
			break
		}
	}
	i := field.lastIdx // points to a time _past_ t

	// out of range: field is 0
	if i-1 < 0 || i >= len(field.points) {
		return [3]float32{0., 0., 0.}
	}
	pt1 := field.points[i-1]
	pt2 := field.points[i]

	dt := pt2.time - pt1.time
	t := float32((time - pt1.time) / (dt)) // 0..1
	B := [3]float32{0, 0, 0}
	for i := range B {
		B[i] = pt1.b[i] + t*(pt2.b[i]-pt1.b[i])
	}
	return B
}


func (s *Sim) ApplyPulse(what string, hz, hy, hx float32, duration, risetime float64) {
	s.apply(what, &pulsedField{[3]float32{hx, hy, hz}, duration, risetime})
	s.Println("Applied "+what+": pulse, (", hx, ", ", hy, ", ", hz, ") T, ", duration, "s FWHM, ", risetime, "s rise- and falltime (0-100%)")
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
	s.Println("Applied "+what+": RF, (", hx, ", ", hy, ", ", hz, ") T, frequency: ", freq, " Hz")
}

type rfField struct {
	b    [3]float32
	freq float64
}

func (field *rfField) GetAppliedField(time float64) [3]float32 {
	sin := float32(math.Sin(field.freq * 2 * math.Pi * time))
	return [3]float32{field.b[X] * sin, field.b[Y] * sin, field.b[Z] * sin}
}

func (s *Sim) ApplyRfRamp(what string, hz, hy, hx float32, freq float64, ramptime float64) {
	s.apply(what, &rfRamp{[3]float32{hx, hy, hz}, freq, ramptime})
	s.Println("Applied "+what+": RF ramp, (", hx, ", ", hy, ", ", hz, ") T, frequency: ", freq, " Hz", "ramp in ", ramptime, "s")
}

type rfRamp struct {
	b    [3]float32
	freq float64
	ramptime float64
}

func (field *rfRamp) GetAppliedField(time float64) [3]float32 {
	fac := 1.
	if time < field.ramptime{
		fac = time/field.ramptime
	}
	sin := float32(math.Sin(field.freq * 2 * math.Pi * time) * fac)
	return [3]float32{field.b[X] * sin, field.b[Y] * sin, field.b[Z] * sin}
}

func (s *Sim) ApplySawtooth(what string, hz, hy, hx float32, freq float64) {
	var st sawtooth
	st = sawtooth(rfField{[3]float32{hx, hy, hz}, freq})
	s.apply(what, &st)
	s.Println("Applied "+what+": sawtooth, (", hx, ", ", hy, ", ", hz, ") T, frequency: ", freq, " Hz")
}

type sawtooth rfField

func (field *sawtooth) GetAppliedField(time float64) [3]float32 {
	sin := float32(math.Asin(math.Sin(field.freq*2*math.Pi*time)) / (math.Pi / 2))
	return [3]float32{field.b[X] * sin, field.b[Y] * sin, field.b[Z] * sin}
}

// Apply a rotating field
func (s *Sim) ApplyRotating(what string, hz, hy, hx float32, freq float64, phaseX, phaseY, phaseZ float64) {
	s.apply(what, &rotatingField{[3]float32{hx, hy, hz}, freq, [3]float64{phaseX, phaseY, phaseZ}})
	s.Println("Applied "+what+": Rotating, (", hx, ", ", hy, ", ", hz, ") T, frequency: ", freq, " Hz", " phases: ", phaseX, ", ", phaseY, ", ", phaseZ, " rad")
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
	s.Println("Applied "+what+": Rotating burst, ", h, " T, frequency: ", freq, " Hz ", "phase between X-Y: ", phase, " risetime: ", risetime, " s", ", duration: ", duration, " s")
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
	norm := float32(0.25 * (math.Erf(time/(field.risetime/2.)-2) + 1) * (2 - math.Erf((time-field.duration)/(field.risetime/2.)) - 1))
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
