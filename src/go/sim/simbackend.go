package sim

func (s *Sim) Cpu() {
	s.backend = CPU
	s.invalidate()
}

func (s *Sim) Gpu() {
	s.backend = GPU
	s.invalidate()
}


// func (s *Sim) Backend(b string) {
//   b = strings.ToLower(b)
//   switch b{
//     default:
//       panic(fmt.Sprint("Unknown backend:", b))
//     case "cpu":
//       s.backend = CPU
//     case "gpu":
//       s.backend = GPU
//   }
//   s.invalidate()
// }
