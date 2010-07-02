package sim

import(
)


// 1st order Euler method
type Euler struct{
  Solver
  
}

func NewEuler(dev Backend, mag *Magnet, dt float) *Euler{
  euler := new(Euler)
  
  euler.Solver = *NewSolver(dev, mag)
  euler.dt = dt
  
  return euler
}

func (this *Euler) Step(){
  m, h := this.m, this.h
  alpha, dt := this.Alpha, this.dt

  this.Normalize(m)
  this.Convolve(m, h)
  this.Torque(m, h, dt/(1+alpha*alpha))
  torque := h

  this.EulerStage(m, torque)
}


// embedding tree:

// Simulation{ ? to avoid typing backend backend backend...(but sim. sim. sim.)
// Euler{
//   Solver{
//     Field{
//       Material;
//       Conv{
//         FFT{
//           size
//           Device{  //sim.Device or cpu.Device
//             // low-level, unsafe simulation primitives
//             NewTensor
//             FFT,
//             Copy,
//             Torque,
//             ...
//           }
//         }
//       }
//     }
//   }
// }
//}
