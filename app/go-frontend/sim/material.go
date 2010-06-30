package sim

import(
  . "math"
  "fmt"
  "io"
)

type Material struct{
  
  AExch   float;      ///< Exchange constant in J/m
  MSat    float;       ///<Saturation magnetization in A/m
  Mu0     float;        ///< Mu0 in N/A^2
  Gamma0  float;     ///< Gyromagnetic ratio in m/As

}

/** All parameters passed in SI units. Program units are used only internally. */
func NewMaterial() *Material{
  
  mat := new(Material);
  mat.Mu0 = 4.0E-7 * Pi;
  mat.Gamma0 = 2.211E5;
  return mat;
}

/** Prints some human-readable information to the screen. */
func (mat *Material) String(out io.Writer) {
  //fmt.Fprintln(out, "Material parameters");
  fmt.Fprintln(out, "AExch      : \t", mat.AExch, " J/m")
  fmt.Fprintln(out, "MSat       : \t", mat.MSat, " A/m")
  fmt.Fprintln(out, "Gamma0      : \t", mat.Gamma0, " m/As")
  fmt.Fprintln(out, "Mu0     : \t", mat.Mu0, " N/A^2")
  fmt.Fprintln(out, "exch length: \t", mat.UnitLength(), " m")
  fmt.Fprintln(out, "unit time  : \t", mat.UnitTime(), " s")
  fmt.Fprintln(out, "unit energy: \t", mat.UnitEnergy(), " J")
  fmt.Fprintln(out, "unit field: \t", mat.UnitField(), " T")
}

/*
 FIELD = Ms;
 LENGTH = sqrt(2.0*A/(Mu0*Ms*Ms));   //2007-02-05: crucial fix: factor sqrt(2), LENGTH is now the exchange length, not just 'a' good length unit.
 TIME = 1.0 / (gamma * Ms);
 ENERGY = A * LENGTH;*/

/** The internal unit of length, expressed in meters. */
func (mat *Material) UnitLength() float{
  assert(mat.Valid());
  return float(Sqrt(2. * float64(mat.AExch / (mat.Mu0*mat.MSat*mat.MSat))));
}

/** The internal unit of time, expressed in seconds. */
func (mat *Material) UnitTime() float{
  assert(mat.Valid());
  return 1.0 / (mat.Gamma0 * mat.MSat);
}

/** The internal unit of field, expressed in tesla. */
func (mat *Material) UnitField() float{
  assert(mat.Valid());
  return mat.Mu0 * mat.MSat;
}

/** The internal unit of energy, expressed in J. */
func (mat *Material) UnitEnergy() float{
  assert(mat.Valid());
  return mat.AExch * mat.UnitLength();
}

func (mat *Material) Valid() bool{
  return mat.AExch != 0. && mat.MSat != 0. && mat.Gamma0 != 0 && mat.Mu0 != 0;
}

func (unit *Material) AssertValid(){
  assert(unit.Valid());
}

