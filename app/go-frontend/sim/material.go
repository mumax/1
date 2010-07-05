package sim

import(
  . "math"
  "fmt"
)

type Material struct{
  
  AExch   float      ///< Exchange constant in J/m
  MSat    float      ///< Saturation magnetization in A/m
  Mu0     float      ///< Mu0 in N/A^2
  Gamma0  float      ///< Gyromagnetic ratio in m/As
  Alpha   float      ///< Damping parameter
  
}


func NewMaterial() *Material{
  
  mat := new(Material);
  mat.Mu0 = 4.0E-7 * Pi;
  mat.Gamma0 = 2.211E5;
  return mat;
}


func (mat *Material) String() string{
  s := "Material:\n"
  s += fmt.Sprintln("AExch      : \t", mat.AExch, " J/m")
  s += fmt.Sprintln("MSat       : \t", mat.MSat, " A/m")
  s += fmt.Sprintln("Gamma0     : \t", mat.Gamma0, " m/As")
  s += fmt.Sprintln("Mu0        : \t", mat.Mu0, " N/A^2")
  s += fmt.Sprintln("exch length: \t", mat.UnitLength(), " m")
  s += fmt.Sprintln("unit time  : \t", mat.UnitTime(), " s")
  s += fmt.Sprintln("unit energy: \t", mat.UnitEnergy(), " J")
  s += fmt.Sprintln("unit field : \t", mat.UnitField(), " T")
  return s
}


//  FIELD = Ms
//  LENGTH = sqrt(2.0*A/(Mu0*Ms*Ms))
//  TIME = 1.0 / (gamma * Ms)
//  ENERGY = A * LENGTH

// The internal unit of length, expressed in meters.
func (mat *Material) UnitLength() float{
  assert(mat.Valid());
  return float(Sqrt(2. * float64(mat.AExch / (mat.Mu0*mat.MSat*mat.MSat))));
}


// The internal unit of time, expressed in seconds.
func (mat *Material) UnitTime() float{
  assert(mat.Valid());
  return 1.0 / (mat.Gamma0 * mat.MSat);
}


// The internal unit of field, expressed in tesla.
func (mat *Material) UnitField() float{
  assert(mat.Valid());
  return mat.Mu0 * mat.MSat;
}


// The internal unit of energy, expressed in J.
func (mat *Material) UnitEnergy() float{
  assert(mat.Valid());
  return mat.AExch * mat.UnitLength();
}


// Returns true if the material parameters are valid
func (mat *Material) Valid() bool{
  return mat.AExch > 0. && mat.MSat > 0. && mat.Gamma0 > 0 && mat.Mu0 > 0
}


func (unit *Material) AssertValid(){
  assert(unit.Valid());
}

