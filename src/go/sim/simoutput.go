package sim

import (
	"fmt"
	"tensor"
	"os"
)

// Sets the output directory where all output files are stored
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


func (s *Sim) Autosave(what, format string, interval float) {
	s.outschedule = s.outschedule[0 : len(s.outschedule)+1]
	output := resolve(what, format)
	output.SetInterval(interval)
	s.outschedule[len(s.outschedule)-1] = output
}


func (s *Sim) Save(what, format string) {
	output := resolve(what, format)
    s.assureMUpToDate()
    output.Save(s)
}

//
type Output interface {
	SetInterval(interval float)
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

func (p *Periodic) SetInterval(interval float) {
	p.period = interval
}

// Takes a text representation of an output (like: "m" "binary") and returns a corresponding output interface.
// No interval is stored yet, so the result can be used for a single save or an interval can be set to use
// it for scheduled output.
func resolve(what, format string) Output {
	switch what {
	default:
		panic("unknown output quantity " + what + ". options are: m")
	case "m":
		switch format {
		default:
			panic("unknown format " + format + ". options are: binary, ascii, png")
		case "binary":
			return &MBinary{&Periodic{0., 0.}}
		case "ascii":
			return &MAscii{&Periodic{0., 0.}}
			//case "png":
		}

	}
	panic("bug")
	return nil // not reached
}

func (m *MAscii) Save(s *Sim) {
	fname := s.outputdir + "/" + "m" + fmt.Sprintf("%06d", s.autosaveIdx) + ".txt"
	out, err := os.Open(fname, os.O_WRONLY|os.O_CREAT, 0666)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(-2)
	}
	defer out.Close()
	tensor.Format(out, s.m)

	m.sinceoutput = s.time
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
