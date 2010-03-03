package sim

import(. "../tensor")

/** 
 * A solver is a "plan" for advancing the magnetization in time. 
 * The solver is initialized with a magnetization pointer and 
 * field plan. solver.Step() advances the magnetization state
 * a little bit in time. 
 *
 * The solver knows its m, i.e., we do not call solver.Step(m)
 * but solver.Step();
 * This is because predictor-corrector solvers remember a few
 * previous m states and should therefore not be applied to 
 * different m pointers.
 */
type Solver interface{
  Step();
  Advance(dt float64);
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

func Torque2(mx, my, mz, hx, hy, hz, alpha, gilbert float) (torquex, torquey, torquez float){
  // - m cross H
     _mxHx := -my * hz + hy * mz;
     _mxHy :=  mx * hz - hx * mz;
     _mxHz := -mx * hy + hx * my;

    // - m cross (m cross H)
     _mxmxHx :=  my * _mxHz - _mxHy * mz;
     _mxmxHy := -mx * _mxHz + _mxHx * mz;
     _mxmxHz :=  mx * _mxHy - _mxHx * my;

    
    torquex = (_mxHx + _mxmxHx * alpha) * gilbert;
    torquey = (_mxHy + _mxmxHy * alpha) * gilbert;
    torquez = (_mxHz + _mxmxHz * alpha) * gilbert;
    
    return;
}

func NormalizeVector2 (mx, my, mz float) (float, float, float){
  invnorm := 1.0 / FSqrt((float64)(mx * mx + my * my + mz * mz));
  return mx * invnorm, my * invnorm, mz * invnorm;
}

func NormalizeVector(m []float){
  invnorm := 1.0 / FSqrt((float64)(m[X] * m[X] + m[Y] * m[Y] + m[Z] * m[Z]));
  m[X] *= invnorm;
  m[Y] *= invnorm;
  m[Z] *= invnorm;  
}
