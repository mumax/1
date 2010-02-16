package sim

import(
  . "../tensor";
  . "math";
  //.  "fmt";
)

type Euler struct{
  m_tensor, h_tensor *Tensor4;
  m, h [][]float;	// components as lists
  field *FieldPlan;
  t, dt float;
}

func NewEuler(m *Tensor4, field *FieldPlan, dt float) *Euler{
  solver := new(Euler);
  
  solver.m_tensor = m;
  solver.h_tensor = NewTensor4(m.Size());
  // store m and h components as lists.
  solver.m, solver.h = make([][]float, 3), make([][]float, 3);
  for c:=0; c<3; c++{
      solver.m[c] = solver.m_tensor.Component(c).List();
      solver.h[c] = solver.h_tensor.Component(c).List();
  }

  solver.t = 0.;
  solver.dt = dt;
  solver.field = field;
  return solver;
}

func (s *Euler) Step(){
    alpha := 1.0;
    gilbert := 1.0 / (1.0 + alpha * alpha);
    
    s.field.Execute(s.m_tensor, s.h_tensor);
    m, h, torque := make([]float, 3), make([]float, 3), make([]float, 3);
    M := s.m;
    H := s.h;
    for i:=0; i<len(s.m[0]); i++{
      for c:=X; c<=Z; c++{
	m[c] = M[c][i];
	h[c] = H[c][i];
      }
      Torque(m, h, torque, alpha, gilbert);
      //Println(torque);
      norm:=0.;
      for c:=X; c<=Z; c++{
	m[c] += s.dt * torque[c];
	norm += m[c]*m[c];
      }
      norm = 1./float(Sqrt(float64(norm)));
      for c:=X; c<=Z; c++{
	M[c][i] = m[c] * norm;
      }
    }
}



func Torque(m, h, torque []float, alpha, gilbert float) {
    // - m cross H
     _mxHx := -m[Y] * h[Z] + h[Y] * m[Z];
     _mxHy := m[X] * h[Z] - h[X] * m[Z];
     _mxHz := -m[X] * h[Y] + h[X] * m[Y];

    // - m cross (m cross H)
     _mxmxHx := m[Y] * _mxHz - _mxHy * m[Z];
     _mxmxHy := -m[X] * _mxHz + _mxHx * m[Z];
     _mxmxHz := m[X] * _mxHy - _mxHx * m[Y];

    
    torque[X] = (_mxHx + _mxmxHx * alpha) * gilbert;
    torque[Y] = (_mxHy + _mxmxHy * alpha) * gilbert;
    torque[Z] = (_mxHz + _mxmxHz * alpha) * gilbert;
}
