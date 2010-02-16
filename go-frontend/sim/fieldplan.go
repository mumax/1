package sim

import(
  . "../tensor";
  . "../fft";
  . "math";
  . "fmt";
)

type FieldPlan struct{
  /** Exchange constant in J/m */
  aExch float;
  /** Saturation magnetization in A/m */
  mSat float;
  /** mu0 in N/A^2 */
  mu0 float;
  /** Gyromagnetic ratio in m/As */
  gamma0 float;

  size []int;
  cellsize []float;

  convPlan *ConvPlan;

  initiated bool;
}

/** All parameters passed in SI units. */
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

func (plan *FieldPlan) Execute(m, h *Tensor4){
  if !plan.initiated { plan.init() }
  plan.convPlan.Execute(m, h);
}

func (plan *FieldPlan) init(){
  
}

func (plan *FieldPlan) PrintInfo() {
  Println("Material parameters");
  Println("aExch      : \t", plan.aExch, " J/m");
  Println("mSat       : \t", plan.mSat, " A/m");
  Println("gamma0      : \t", plan.gamma0, " m/As");
  Println("mu0     : \t", plan.mu0, " N/A^2");
  Println("exch length: \t", plan.UnitLength(), " m");
  Println("unit time  : \t", plan.UnitTime(), " s");
  Println("unit energy: \t", plan.UnitEnergy(), " J");
  Println("Geometry");
  Println("Grid size  : \t", plan.Size());
  Print("Cell size  : \t");
  for i:=range(plan.cellsize){
    Print(plan.UnitLength() * plan.cellsize[i], " ");
  }
  Print("(m), (");
   for i:=range(plan.cellsize){
    Print(plan.cellsize[i], " ");
  }
  Println("exch. lengths)");

  Print("Sim size   : \t ");
  for i:=range(plan.size){
    Print(float(plan.Size()[i]) * plan.UnitLength() * plan.cellsize[i], " ");
  }
  Println("(m)");
}

/*
 FIELD = Ms;
 LENGTH = sqrt(2.0*A/(mu0*Ms*Ms));   //2007-02-05: crucial fix: factor sqrt(2), LENGTH is now the exchange length, not just 'a' good length unit.
 TIME = 1.0 / (gamma * Ms);
 ENERGY = A * LENGTH;*/

func (plan *FieldPlan) UnitLength() float{
  return float(Sqrt(2. * float64(plan.aExch / (plan.mu0*plan.mSat*plan.mSat))));
}


func (plan *FieldPlan) UnitTime() float{
  return 1.0 / (plan.gamma0 * plan.mSat);
}


func (plan *FieldPlan) UnitField() float{
  return plan.mSat;
}


func (plan *FieldPlan) UnitEnergy() float{
  return plan.aExch * plan.UnitLength();
}

func (plan *FieldPlan) Size() []int{
  return plan.size;
}
