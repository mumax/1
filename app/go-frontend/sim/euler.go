package sim

import(
)


// 1st order Euler method
type Euler struct{
  Dt float
  Field
}

func(this *Euler) String() string{
  return "Euler" + this.Field.String() + "--\n"
}

func NewEuler(dev Backend, mag *Magnet, Dt float) *Euler{
  this := new(Euler)
  this.Field = *NewField(dev, mag)
  this.Dt = Dt
  return this
}

func (this *Euler) Step(){
  Debugvv( "Euler.Step()" )
  m, h := this.m, this.h
  alpha, Dt := this.Alpha, this.Dt

  this.Normalize(m)
  this.CalcHeff(this.m, this.h)
  this.DeltaM(m, h, alpha, Dt/(1+alpha*alpha))
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
