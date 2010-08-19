package sim

import (
	"fmt"
	"tensor"
	"os"
)

func (s *Sim) OutputDir(outputdir string) {
	// remove trailing slash if present
	//   if outputdir[len(outputdir)-1] == '/'{
	//     outputdir = outputdir[0:len(outputdir)-1]
	//   }
	err := os.Mkdir(outputdir, 0777)
	if err != nil {
		fmt.Fprintln(os.Stderr, err) // we should not abort here, file exists is a possible error
	}
	s.outputdir = outputdir
	//does not invalidate
}



type Output interface{
  NeedSave(time float) bool
  Save(sim *Sim)
}


type Periodic struct{
  period float
  sinceoutput float
}

func (p *Periodic) NeedSave(time float) bool{
  return time - p.sinceoutput >= p.period
}

type MOutput struct{
  *Periodic
}

func (m *MOutput) Save(s *Sim){
  fname := s.outputdir + "/" + "m" + fmt.Sprintf("%06d", s.autosaveIdx) + ".t"
  tensor.WriteFile(fname, s.m)
  m.sinceoutput = s.time
}


func (s *Sim) Autosave(what string, interval float){
  //TODO other than m
  s.outschedule = s.outschedule[0:len(s.outschedule)+1]
  s.outschedule[len(s.outschedule)-1] = &MOutput{&Periodic{interval, 0.}}
}

// func (s *Sim) autosavem() {
// 	s.autosaveIdx++ // we start at 1 to stress that m0 has not been saved
// 	TensorCopyFrom(s.solver.M(), s.m)
// 	fname := s.outputdir + "/" + "m" + fmt.Sprintf("%06d", s.autosaveIdx) + ".t"
// 	tensor.WriteFile(fname, s.m)
// }

// func (s *Sim) AutosaveM(interval float) {
// 	s.savem = interval
// 	//does not invalidate
// }
