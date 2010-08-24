package sim

import ()


// 1st order Euler method
type Euler struct {
	SolverState
}

func (this *Euler) String() string {
	return "Euler\n" + this.Field.String() + "--\n"
}

func NewEuler(dev Backend, field *Field, Dt float) *Euler { //todo: do not need backend param
	this := new(Euler)
	this.Field = field
	this.Dt = Dt
	return this
}

func (this *Euler) Step() {
	Debugvv("Euler.Step()")
	//TODO normalize m before first step (sim.go)
	//TODO make them private!
	m, h := this.m, this.h
	alpha, dt := this.Alpha, this.Dt

	// 	this.Normalize(this.m)
	this.CalcHeff(m, h)
	this.DeltaM(m, h, alpha, dt/(1+alpha*alpha))
	deltaM := h // h is overwritten by deltaM

	this.Add(m, deltaM)
	this.Normalize(m)
}


// embedding tree:

// Simulation{ ? to avoid typing backend backend backend...(but sim. sim. sim.)
// Euler{
//   TimeStep{
//     Field{
//       Magnet{
//         Material
//         Size
//       }
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
