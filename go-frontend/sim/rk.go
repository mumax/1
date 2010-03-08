package sim

/*
 * The classical 4th order Runge-Kutta method.
 */

import(
  "fmt";
  "os";
  . "math";
  . "../tensor";
)

// "constants"
var NAN = float(NaN());

// RK4
var butcher4 = [...][4]float{
  [4]float{NAN, NAN, NAN, NAN},
  [4]float{1.0 / 2.0, NAN, NAN},
  [4]float{0.0, 1.0 / 2.0, NAN},
  [4]float{0.0, 0.0, 1.0, NAN}
};

var      h4 = [...]float{0.0, 1.0 / 2.0, 1.0 / 2.0, 1.0};
var weight4 = [...]float{1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0};

/**
 * The classic 4th-order Runge-Kutta method.
 */
type RK4 struct{
  m *Tensor4;
  h *Tensor4;
  m0 *Tensor4; // backup of m
  field *FieldPlan;
  k [4]*Tensor4; // [4] // Idea: cache efficiency: make [X,Y,Z] the last component?
  dt float;
  t float64;
  maxDm float;
}

func NewRK4Fixed(m *Tensor4, field *FieldPlan, dt float) *RK4{
  return newRK4(m , field, dt, 0.0);   
}


func NewRK4MaxDm(m *Tensor4, field *FieldPlan, maxDm float) *RK4{
  return newRK4(m, field, 0.0, maxDm);
}


func newRK4(m *Tensor4, field *FieldPlan, dt, maxDm float) *RK4{
  rk4 := new(RK4);

  rk4.m = m;
  rk4.h = NewTensor4(m.Size());
  rk4.m0 = NewTensor4(m.Size());
  rk4.field = field;
  for i:=range(rk4.k){
    rk4.k[i] = NewTensor4(m.Size());
  }
  rk4.dt = dt;
  rk4.maxDm = maxDm;
  
  return rk4;
}


func (rk4 *RK4) Step(){
    
    alpha := 1.0;
    gilbert := 1.0 / (1.0 + alpha * alpha);
    
    M := rk4.m;
    Mx := M.Component(X).List(); // idea: could/should be moved outside Step()
    My := M.Component(Y).List();
    Mz := M.Component(Z).List();
    
    H := rk4.h;
    Hx := H.Component(X).List();
    Hy := H.Component(Y).List();
    Hz := H.Component(Z).List();
    
    var Kx, Ky, Kz [4][]float;
    
    for i:=range(weight4){
      Kx[i] = rk4.k[i].Component(X).List();
      Ky[i] = rk4.k[i].Component(Y).List();
      Kz[i] = rk4.k[i].Component(Z).List();
    }
    
    // initial RK4:
    //	backup m
    CopyTo(rk4.m, rk4.m0); // idea: include in following loop for caching.
    
    rk4.field.Execute(rk4.m, rk4.h);
    
    var maxTorque float;
    for c:=0; c<len(Mx); c++{
      tx, ty, tz := Torque2(Mx[c], My[c], Mz[c], Hx[c], Hy[c], Hz[c], alpha, gilbert);
      Kx[0][c], Ky[0][c], Kz[0][c] = tx, ty, tz;
      torque2 := tx*tx + ty*ty + tz*tz;
      if torque2 > maxTorque{
	maxTorque = torque2;
      }
    }
    maxTorque = FSqrt(float64(maxTorque));
    if rk4.maxDm != 0.{
	rk4.dt = rk4.maxDm / maxTorque;
	fmt.Fprintln(os.Stderr, "maxTorque:", maxTorque);
	fmt.Fprintln(os.Stderr, "dt:", rk4.dt);
     }
    
    dt := rk4.dt;
    
    // run over butcher tableau:
    for i:=1; i<len(weight4); i++{
      // reset m
      CopyTo(rk4.m0, rk4.m); // idea: include in following loop for better cache use
      // new m:
      for c:=range(Mx){	  
	for j:=0; j<i; j++{
	  Mx[c] += dt * butcher4[i][j] * Kx[i-1][c];
	  My[c] += dt * butcher4[i][j] * Ky[i-1][c];
	  Mz[c] += dt * butcher4[i][j] * Kz[i-1][c];
	}
	Mx[c], My[c], Mz[c] = NormalizeVector2(Mx[c], My[c], Mz[c]);
      }
   
      rk4.field.Execute(rk4.m, rk4.h);
    
      // new k
      for c:= range(Mx){	
	Kx[i][c], Ky[i][c], Kz[i][c] = Torque2(Mx[c], My[c], Mz[c], Hx[c], Hy[c], Hz[c], alpha, gilbert);
      }
    }
    
     //new m:
     // reset first:
     CopyTo(rk4.m0, rk4.m); // idea: include in following loop for caching.
     for c:=range(Mx){
       for i:=0; i<len(weight4); i++ {
	Mx[c] += dt * weight4[i] * Kx[i][c];
	My[c] += dt * weight4[i] * Ky[i][c];
	Mz[c] += dt * weight4[i] * Kz[i][c];
       }
       Mx[c], My[c], Mz[c] = NormalizeVector2(Mx[c], My[c], Mz[c]);
     }
     
     
}
