package sim

import(
  . "../tensor";
  . "../fft";
  . "math";
  "fmt";
)

/**
 * A FieldPlan takes a magnetization m and returns an effective field H:
 * plan.Execute(m, H);
 * Internally, it uses a convPlan to do the convolution. The field plan
 * sets up the kernel for the convPlan, based on material parameters and
 * cell size.
 */
type FieldPlan struct{
  /** Exchange constant in J/m */
  aExch float;
  /** Saturation magnetization in A/m */
  mSat float;
  /** mu0 in N/A^2 */
  mu0 float;
  /** Gyromagnetic ratio in m/As */
  gamma0 float;
  /** Mesh size, e.g. 64x64x4 */
  size []int;
  /** Cell size in exchange lengths, e.g. 0.5x0.5.1.2 */
  cellsize []float;

  convPlan *ConvPlan;
  //initiated bool;
}

/** All parameters passed in SI units. Program units are used only internally. */
func NewFieldPlan(size []int, cellsize []float, mSat, aExch float) *FieldPlan{
  plan := new(FieldPlan);
  
  plan.aExch = aExch;
  plan.mSat = mSat;
  plan.mu0 = 4.0E-7 * Pi;
  plan.gamma0 = 2.211E5;
  
  plan.size = size;
  for c:=range(cellsize){
    cellsize[c] /= plan.UnitLength();
  }
  plan.cellsize = cellsize;

  //size3D := []int{3, size[0], size[1], size[2]};
  DemagKernel := FaceKernel(size, cellsize);
  ExchKernel := Exch6NgbrKernel(size, cellsize);
  Kernel := Add(DemagKernel, ExchKernel);
  plan.convPlan = NewConvPlan(size, Kernel);
  PrintVectors(FOpen("/home/arne/Desktop/Kx"), Slice(Kernel, 0, 0));
  return plan;
}

/** Executes the plan: stores the field h corresponding to the magnetization m. */
func (plan *FieldPlan) Execute(m, h *Tensor4){
  //if !plan.initiated { plan.init() }
  plan.convPlan.Execute(m, h);
}

// func (plan *FieldPlan) init(){
//   
// }

/** Prints some human-readable information to the screen. */
func (plan *FieldPlan) PrintInfo() {
  fmt.Println("Material parameters");
  fmt.Println("aExch      : \t", plan.aExch, " J/m");
  fmt.Println("mSat       : \t", plan.mSat, " A/m");
  fmt.Println("gamma0      : \t", plan.gamma0, " m/As");
  fmt.Println("mu0     : \t", plan.mu0, " N/A^2");
  fmt.Println("exch length: \t", plan.UnitLength(), " m");
  fmt.Println("unit time  : \t", plan.UnitTime(), " s");
  fmt.Println("unit energy: \t", plan.UnitEnergy(), " J");
  fmt.Println("Geometry");
  fmt.Println("Grid size  : \t", plan.Size());
  fmt.Print("Cell size  : \t");
  for i:=range(plan.cellsize){
    fmt.Print(plan.UnitLength() * plan.cellsize[i], " ");
  }
  fmt.Print("(m), (");
   for i:=range(plan.cellsize){
    fmt.Print(plan.cellsize[i], " ");
  }
  fmt.Println("exch. lengths)");

  fmt.Print("Sim size   : \t ");
  for i:=range(plan.size){
    fmt.Print(float(plan.Size()[i]) * plan.UnitLength() * plan.cellsize[i], " ");
  }
  fmt.Println("(m)");
}

/*
 FIELD = Ms;
 LENGTH = sqrt(2.0*A/(mu0*Ms*Ms));   //2007-02-05: crucial fix: factor sqrt(2), LENGTH is now the exchange length, not just 'a' good length unit.
 TIME = 1.0 / (gamma * Ms);
 ENERGY = A * LENGTH;*/

/** The internal unit of length, expressed in meters. */
func (plan *FieldPlan) UnitLength() float{
  return float(Sqrt(2. * float64(plan.aExch / (plan.mu0*plan.mSat*plan.mSat))));
}

/** The internal unit of time, expressed in seconds. */
func (plan *FieldPlan) UnitTime() float{
  return 1.0 / (plan.gamma0 * plan.mSat);
}

/** The internal unit of field, expressed in A/m. */
func (plan *FieldPlan) UnitField() float{
  return plan.mSat;
}

/** The internal unit of energy, expressed in J. */
func (plan *FieldPlan) UnitEnergy() float{
  return plan.aExch * plan.UnitLength();
}

/** The plan's mesh size. */
func (plan *FieldPlan) Size() []int{
  return plan.size;
}
