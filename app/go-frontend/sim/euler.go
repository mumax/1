package sim

import(
)


// 1st order Euler method
type Euler struct{
  dt float
  Field
}

func(this *Euler) String() string{
  return "Euler" + this.Field.String() + "--\n"
}

func NewEuler(dev Backend, mag *Magnet, dt float) *Euler{
  this := new(Euler)
  this.Field = *NewField(dev, mag)
  this.dt = dt
  return this
}

func (this *Euler) Step(){
  Debugvv( "Euler.Step()" )
  m, h := this.m, this.h
  alpha, dt := this.Alpha, this.dt

  this.Normalize(m)
  this.CalcHeff(this.m, this.h)
  this.DeltaM(m, h, alpha, dt/(1+alpha*alpha))
  deltaM := h // h is overwritten by deltaM

  this.Add(m, deltaM)
  this.Normalize(m)
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
