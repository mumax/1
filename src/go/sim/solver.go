package sim
//
// import(
//   "tensor"
//   "fmt"
// )
//
// // Solver contains common code of all concrete solvers,
// // who embed it
// // TODO perhaps we should pull m, h up so they can stay when the solver is changed for another one
// type Solver struct{
//
//   dt float
//   Field
// }
//
//
// func NewSolver(dev Backend, mag *Magnet) *Solver{
//   solver := new(Solver)
//
//   solver.m = NewTensor(dev, Size4D(mag.Size()))
//   solver.h = NewTensor(dev, Size4D(mag.Size()))
//   solver.Field = *NewField(dev, mag)
//
//   return solver
// }
//

//
// // // TODO do not pass alpha
// // func(s *Solver) Torque(m, h *Tensor, dtGilbert float){
// //   Debugvv( "Solver.Torque()" )
// //   assert(len(m.size) == 4)
// //   assert(tensor.EqualSize(m.size, h.size))
// //
// //   N := m.size[1] * m.size[2] * m.size[3]
// //   s.deltaM(m.data, h.data, s.Alpha, dtGilbert, N)
// // }
//
//
//
//
//
// // func(s *Solver) EulerStage(m, torque *Tensor){
// //   Debugvv( "Solver.EulerStage()" )
// //   assert(len(m.size) == 4)
// //   assert(tensor.EqualSize(m.size, torque.size))
// //
// //   N := m.size[1] * m.size[2] * m.size[3]
// //   s.add(m.data, torque.data, N)
// //
// // }
//
// func(s *Solver) String() string{
//   str := "Solver:\n"
//   str += fmt.Sprintln("dt:", s.dt * s.UnitTime(), "s")
//   str += s.Field.String()
//   str += "--\n"
//   return str
// }
