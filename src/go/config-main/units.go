package main

import (
	. "math"
	"fmt"
	"io"
)

/**
 * A Units takes a magnetization m and returns an effective field H:
 * Units.Execute(m, H);
 * Internally, it uses a convPlan to do the convolution. The field Units
 * sets up the kernel for the convPlan, based on material parameters and
 * cell Size.
 */
type Units struct {
	/** Exchange constant in J/m */
	AExch float
	/** Saturation magnetization in A/m */
	MSat float
	/** Mu0 in N/A^2 */
	Mu0 float
	/** Gyromagnetic ratio in m/As */
	Gamma0 float
	/** Mesh Size, e.g. 64x64x4 */
	Size []int
	/** Cell Size in exchange lengths, e.g. 0.5x0.5.1.2 */
	CellSize []float
}

/** All parameters passed in SI units. Program units are used only internally. */
func NewUnits() *Units {

	units := new(Units)

	units.Mu0 = 4.0E-7 * Pi
	units.Gamma0 = 2.211E5

	return units
}

/** Prints some human-readable information to the screen. */
func (units *Units) PrintInfo(out io.Writer) {
	fmt.Fprintln(out, "Material parameters")
	fmt.Fprintln(out, "AExch      : \t", units.AExch, " J/m")
	fmt.Fprintln(out, "MSat       : \t", units.MSat, " A/m")
	fmt.Fprintln(out, "Gamma0      : \t", units.Gamma0, " m/As")
	fmt.Fprintln(out, "Mu0     : \t", units.Mu0, " N/A^2")
	fmt.Fprintln(out, "exch length: \t", units.UnitLength(), " m")
	fmt.Fprintln(out, "unit time  : \t", units.UnitTime(), " s")
	fmt.Fprintln(out, "unit energy: \t", units.UnitEnergy(), " J")
	fmt.Fprintln(out, "unit field: \t", units.UnitField(), " T")
	fmt.Fprintln(out, "Geometry")
	fmt.Fprintln(out, "Grid Size  : \t", units.Size)
	fmt.Fprint(out, "Cell Size  : \t")
	for i := range units.CellSize {
		fmt.Fprint(out, units.UnitLength()*units.CellSize[i], " ")
	}
	fmt.Fprint(out, "(m), (")
	for i := range units.CellSize {
		fmt.Fprint(out, units.CellSize[i], " ")
	}
	fmt.Fprintln(out, "exch. lengths)")

	fmt.Fprint(out, "Sim Size   : \t ")
	for i := range units.Size {
		fmt.Fprint(out, float(units.Size[i])*units.UnitLength()*units.CellSize[i], " ")
	}
	fmt.Fprintln(out, "(m)")
}

/*
 FIELD = Ms;
 LENGTH = sqrt(2.0*A/(Mu0*Ms*Ms));   //2007-02-05: crucial fix: factor sqrt(2), LENGTH is now the exchange length, not just 'a' good length unit.
 TIME = 1.0 / (gamma * Ms);
 ENERGY = A * LENGTH;*/

/** The internal unit of length, expressed in meters. */
func (units *Units) UnitLength() float {
	assert(units.Valid())
	return float(Sqrt(2. * float64(units.AExch/(units.Mu0*units.MSat*units.MSat))))
}

/** The internal unit of time, expressed in seconds. */
func (units *Units) UnitTime() float {
	assert(units.Valid())
	return 1.0 / (units.Gamma0 * units.MSat)
}

/** The internal unit of field, expressed in tesla. */
func (units *Units) UnitField() float {
	assert(units.Valid())
	return units.Mu0 * units.MSat
}

/** The internal unit of energy, expressed in J. */
func (units *Units) UnitEnergy() float {
	assert(units.Valid())
	return units.AExch * units.UnitLength()
}

func (units *Units) Valid() bool {
	return units.AExch != 0. && units.MSat != 0. && units.Gamma0 != 0 && units.Mu0 != 0
}

func (unit *Units) AssertValid() {
	assert(unit.Valid())
}
