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


type Output interface {
	NeedSave(time float) bool
	Save(sim *Sim)
}


type Periodic struct {
	period      float
	sinceoutput float
}

func (p *Periodic) NeedSave(time float) bool {
	return time-p.sinceoutput >= p.period
}

type MBinary struct {
	*Periodic
}

func (m *MBinary) Save(s *Sim) {
	fname := s.outputdir + "/" + "m" + fmt.Sprintf("%06d", s.autosaveIdx) + ".t"
	tensor.WriteFile(fname, s.m)
	m.sinceoutput = s.time
}

type MAscii struct {
	*Periodic
}

func (m *MAscii) Save(s *Sim) {
	fname := s.outputdir + "/" + "m" + fmt.Sprintf("%06d", s.autosaveIdx) + ".txt"
	out, err := os.Open(fname, os.O_WRONLY | os.O_CREAT, 0666)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(-2)
	}
	defer out.Close()
	tensor.Format(out, s.m)

	m.sinceoutput = s.time
}


func (s *Sim) Autosave(what, format string, interval float) {
	s.outschedule = s.outschedule[0 : len(s.outschedule)+1]
	switch what {
	case "m":
		switch format {
		case "binary":
			s.outschedule[len(s.outschedule)-1] = &MBinary{&Periodic{interval, 0.}}
		case "ascii":
			s.outschedule[len(s.outschedule)-1] = &MAscii{&Periodic{interval, 0.}}
		case "png":
		default:
			panic("unknown format " + format + ". options are: binary, ascii, png")
		}
	default:
		panic("unknown output quantity " + what + ". options are: m")
	}
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
