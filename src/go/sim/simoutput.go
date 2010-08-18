package sim

import (
	"fmt"
	"tensor"
	"os"
)

func (s *Sim) OutputDir(outputdir string){
  // remove trailing slash if present
//   if outputdir[len(outputdir)-1] == '/'{
//     outputdir = outputdir[0:len(outputdir)-1]
//   }
  err := os.Mkdir(outputdir, 0777)
  if err != nil{
    fmt.Fprintln(os.Stderr, err)
    // we should not abort here, file exists is a possible error
  }
  s.outputdir = outputdir
  //does not invalidate
}

func (s *Sim) autosavem() {
	s.autosaveIdx++ // we start at 1 to stress that m0 has not been saved
	TensorCopyFrom(s.solver.M(), s.m)
	fname := s.outputdir + "/" + "m" + fmt.Sprintf("%06d", s.autosaveIdx) + ".t"
	tensor.WriteFile(fname, s.m)
}

func (s *Sim) AutosaveM(interval float) {
	s.savem = interval
	//does not invalidate
}
