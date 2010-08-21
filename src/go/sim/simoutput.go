package sim

// This file implements the methods for scheduling output

import (
	"fmt"
	"tensor"
	"os"
	"io"
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

// Schedules a quantity for autosave
// E.g.: "autosave m binary 1E-9" will save the magnetization in binary format every ns
func (s *Sim) Autosave(what, format string, interval float) {
	s.outschedule = s.outschedule[0 : len(s.outschedule)+1]
	output := resolve(what, format)
	output.SetInterval(interval)
	s.outschedule[len(s.outschedule)-1] = output
}

// Saves a quantity just once
// E.g.: "save m binary" saves the current magnetization state
func (s *Sim) Save(what, format string) {
	output := resolve(what, format)
	s.assureMUpToDate()
	output.Save(s)
}


//____________________________________________________________________ internal

// INTERNAL: Entries in the list of scheduled output have this interface
type Output interface {
	// Set the autosave interval in seconds
	SetInterval(interval float)
	// Returns true if the output needs to saved at this time
	NeedSave(time float) bool
	// After NeedSave() returned true, the simulation will make sure the local copy of m is up to date and the autosaveIdx gets updated. Then Save() is called to save the output
	Save(sim *Sim)
}

// INTERNAL: Common superclass for all periodic outputs
type Periodic struct {
	period      float
	sinceoutput float
}

// INTERNAL
func (p *Periodic) NeedSave(time float) bool {
	return time-p.sinceoutput >= p.period
}

// INTERNAL
func (p *Periodic) SetInterval(interval float) {
	p.period = interval
}

// INTERNAL
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
		case "png":
			return &MPng{&Periodic{0., 0.}}
		}
	case "table":
      //format gets ignored for now
      return &Table{&Periodic{0., 0.}, nil}
	}

	panic("bug")
	return nil // not reached
}

//__________________________________________ ascii


// TODO: it would be nice to have a separate date sturcture for the format and one for the data.
// with a better input file parser we could allow any tensor to be stored:
// save average(component(m, z)) jpg
// INTERNAL
type MAscii struct {
	*Periodic
}

// INTERNAL
func (m *MAscii) Save(s *Sim) {
	fname := s.outputdir + "/" + "m" + fmt.Sprintf("%06d", s.autosaveIdx) + ".txt"
	out, err := os.Open(fname, os.O_WRONLY|os.O_CREAT, 0666)
	defer out.Close()
	if err != nil {
		panic(err)
	}
	tensor.Format(out, s.m)
	m.sinceoutput = s.time
}

type Table struct {
	*Periodic
	out io.Writer
}

const TABLE_HEADER = "# id\t time\t mx\t my\t mz"

func (t *Table) Save(s *Sim) {
	if t.out == nil {
		fname := s.outputdir + "/" + "datatable.txt"
		out, err := os.Open(fname, os.O_WRONLY|os.O_CREAT, 0666)
		// todo: out is not closed
		if err != nil {
			panic(err)
		}
		t.out = out
		Debugv("Opened data table file")
		fmt.Fprintln(out, TABLE_HEADER)
	}
	mx, my, mz := m_average(s.m)
	fmt.Fprintln(t.out, s.autosaveIdx, s.time, mx, my, mz)
}

func m_average(m *tensor.Tensor4) (mx, my, mz float) {
	count := 0
	a := m.Array()
	for i := range a[0] {
		for j := range a[0][i] {
			for k := range a[0][i][j] {
				mx += a[X][i][j][k]
				my += a[Y][i][j][k]
				mz += a[Z][i][j][k]
				count++
			}
		}
	}
	mx /= float(count)
	my /= float(count)
	mz /= float(count)
	return
}

//_________________________________________ binary

// INTERNAL
type MBinary struct {
	*Periodic
}

// INTERNAL
func (m *MBinary) Save(s *Sim) {
	fname := s.outputdir + "/" + "m" + fmt.Sprintf("%06d", s.autosaveIdx) + ".t"
	tensor.WriteFile(fname, s.m)
	m.sinceoutput = s.time
}


//_________________________________________ png

// INTERNAL
type MPng struct {
	*Periodic
}

// INTERNAL
func (m *MPng) Save(s *Sim) {
	fname := s.outputdir + "/" + "m" + fmt.Sprintf("%06d", s.autosaveIdx) + ".png"
	out, err := os.Open(fname, os.O_WRONLY|os.O_CREAT, 0666)
	defer out.Close()
	if err != nil {
		panic(err)
	}
	PNG(out, s.m)
	m.sinceoutput = s.time
}
