//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

// This file implements the methods for scheduling output

// TODO each kind of output should be scheduled only once
// autosave m 1E-9; autosave m 2E-9
// should remove the first entry.
// TODO scheduling a table twice will wipe the previous content...

import (
	"fmt"
	"tensor"
	"os"
	// 	"io"
	"tabwriter"
)

// Sets the output directory where all output files are stored
func (s *Sim) outputDir(outputdir string) {
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
// We use SI units! So that the autosave information is independent of the material parameters!
// E.g.: "autosave m binary 1E-9" will save the magnetization in binary format every ns
func (s *Sim) Autosave(what, format string, interval float) {
	// interval in SI units
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
	// Set the autosave interval in seconds - SI units!
	SetInterval(interval float)
	// Returns true if the output needs to saved at this time - SI units!
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
	return time == 0. || time-p.sinceoutput >= p.period
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

const FILENAME_FORMAT = "%08d"

// INTERNAL
func (m *MAscii) Save(s *Sim) {
	fname := s.outputdir + "/" + "m" + fmt.Sprintf(FILENAME_FORMAT, s.autosaveIdx) + ".txt"
	out, err := os.Open(fname, os.O_WRONLY|os.O_CREAT, 0666)
	defer out.Close()
	if err != nil {
		panic(err)
	}
	tensor.Format(out, s.mLocal)
	m.sinceoutput = float(s.time) * s.UnitTime()
}

type Table struct {
	*Periodic
	out *tabwriter.Writer
}

// table output settings
const (
	TABLE_HEADER = "# time (s)\t mx\t my\t mz\t Bx\t By\t Bz\tdt(s)\terror\tid"
	COL_WIDTH    = 15
)

func (t *Table) Save(s *Sim) {
	if t.out == nil {
		fname := s.outputdir + "/" + "datatable.txt"
		out, err := os.Open(fname, os.O_WRONLY|os.O_CREAT|os.O_TRUNC, 0666)
		// todo: out is not closed
		if err != nil {
			panic(err)
		}
		t.out = tabwriter.NewWriter(out, COL_WIDTH, 4, 0, ' ', 0)
		s.Println("Opened data table file")
		fmt.Fprintln(t.out, TABLE_HEADER)
	}
	mx, my, mz := m_average(s.mLocal)
	B := s.UnitField()
	fmt.Fprintf(t.out, "%e\t% f\t% f\t% f\t", float(s.time)*s.UnitTime(), mx, my, mz)
	fmt.Fprintf(t.out, "% g\t% g\t% g\t", s.hext[X]*B, s.hext[Y]*B, s.hext[Z]*B)
	fmt.Fprintf(t.out, "%.5g\t", s.dt*s.UnitTime())
	fmt.Fprintf(t.out, "%.4g\t", s.stepError)
	fmt.Fprintf(t.out, FILENAME_FORMAT, s.autosaveIdx)
	fmt.Fprintln(t.out)
	t.out.Flush()
	t.sinceoutput = float(s.time) * s.UnitTime()
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
	fname := s.outputdir + "/" + "m" + fmt.Sprintf(FILENAME_FORMAT, s.autosaveIdx) + ".t"
	tensor.WriteFile(fname, s.mLocal)
	m.sinceoutput = float(s.time) * s.UnitTime()
}


//_________________________________________ png

// INTERNAL
type MPng struct {
	*Periodic
}

// INTERNAL
func (m *MPng) Save(s *Sim) {
	fname := s.outputdir + "/" + "m" + fmt.Sprintf(FILENAME_FORMAT, s.autosaveIdx) + ".png"
	out, err := os.Open(fname, os.O_WRONLY|os.O_CREAT, 0666)
	defer out.Close()
	if err != nil {
		panic(err)
	}
	PNG(out, s.mLocal)
	m.sinceoutput = float(s.time) * s.UnitTime()
}
