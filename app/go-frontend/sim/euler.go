package gpu

import(
)

type Euler struct{
  Solver
  
}

func NewEuler(field *Field, alpha, dt float){
  euler := new(Euler)
  euler.Field = *field
  euler.alpha = alpha
  euler.dt = dt
}

func (this *Euler) Step(){
  m, h := this.m, this.h
  alpha, dt := this.alpha, this.dt
  
  this.Convolve(m, h)
  Torque(m, h, alpha, dt/(1+alpha*alpha))
  torque := h

  EulerStage(m, torque)
}


// embedding tree:

// Euler{
//   Solver{
//     Field{
//       Material;
//       Conv{
//         FFT{
//           size
//           Device{  //gpu.Device or cpu.Device
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

