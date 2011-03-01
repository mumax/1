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
	. "mumax/common"
	"fmt"
	"mumax/tensor"
	"os"
	"mumax/draw"
	"mumax/omf"
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
func (s *Sim) Autosave(what, format string, interval float32) {
	// interval in SI units
	s.outschedule = s.outschedule[0 : len(s.outschedule)+1]
	output := resolve(what, format)
	output.SetInterval(interval)
	s.outschedule[len(s.outschedule)-1] = output
}

// Saves a quantity just once
// E.g.: "save m binary" saves the current magnetization state
func (s *Sim) Save(what, format string) {
	s.init() // We must init() so m gets Normalized etc...
	output := resolve(what, format)
	s.assureOutputUpToDate()
	output.Save(s)
}


//____________________________________________________________________ internal

// INTERNAL: Entries in the list of scheduled output have this interface
type Output interface {
	// Set the autosave interval in seconds - SI units!
	SetInterval(interval float32)
	// Returns true if the output needs to saved at this time - SI units!
	NeedSave(time float32) bool
	// After NeedSave() returned true, the simulation will make sure the local copy of m is up to date and the autosaveIdx gets updated. Then Save() is called to save the output
	Save(sim *Sim)
}

// INTERNAL: Common superclass for all periodic outputs
type Periodic struct {
	period      float32
	sinceoutput float32
}

// INTERNAL
func (p *Periodic) NeedSave(time float32) bool {
	return time == 0. || time-p.sinceoutput >= p.period
}

// INTERNAL
func (p *Periodic) SetInterval(interval float32) {
	p.period = interval
}

// INTERNAL
// Takes a text representation of an output (like: "m" "binary") and returns a corresponding output interface.
// No interval is stored yet, so the result can be used for a single save or an interval can be set to use
// it for scheduled output.
func resolve(what, format string) Output {
	switch what {
	default:
		panic("unknown output quantity " + what + ". options are: m, table")
	case "m":
		switch format {
		default:
			panic("unknown format " + format + ". options are: binary (=binary4,omf), text (=ascii), png")
			//		case "binary":
			//			return &MBinary{&Periodic{0., 0.}}
			//		case "ascii":
			//			return &MAscii{&Periodic{0., 0.}}
		case "png":
			return &MPng{&Periodic{0., 0.}}
		case "binary", "binary4", "omf":
			return &MOmf{&Periodic{0., 0.}, "binary"}
		case "text", "ascii":
			return &MOmf{&Periodic{0., 0.}, "text"}
		}
	case "table":
		//format gets ignored for now
		return &Table{&Periodic{0., 0.}}
	case "torque":
		return &Torque{&Periodic{0., 0.}, format}
	}
	panic("bug")
	return nil // not reached
}

// How to format the id for the file name, e.g.: m000000001.omf
const FILENAME_FORMAT = "%08d"


type Table struct {
	*Periodic
}

func (s *Sim) initTabWriter() {
	if s.tabwriter != nil {
		panic(Bug("Tabwriter already initiated."))
	}
	fname := s.outputdir + "/" + "datatable.odt"
	out := MustOpenWRONLY(fname)
	s.tabwriter = omf.NewTabWriter(out)
	s.tabwriter.AddColumn("Time", "s")
	s.tabwriter.AddColumn("Mx/Ms", "")
	s.tabwriter.AddColumn("My/Ms", "")
	s.tabwriter.AddColumn("Mz/Ms", "")
	s.tabwriter.AddColumn("Bx", "T")
	s.tabwriter.AddColumn("By", "T")
	s.tabwriter.AddColumn("Bz", "T")
	s.tabwriter.AddColumn("max_dm/dt", "gammaMs")
	s.tabwriter.AddColumn("min_Mz/Ms", "")
	s.tabwriter.AddColumn("max_Mz/Ms", "")
	s.tabwriter.AddColumn("id", "")
}

func (t *Table) Save(s *Sim) {
	if s.tabwriter == nil {
		s.initTabWriter()
	}

	// calculate reduced quantities
	m := [3]float32{}
	torque := [3]float32{}
	//N := Len(s.size3D)
	for i := range m {
		m[i] = s.devsum.Reduce(s.mDev.comp[i]) / s.avgNorm
		torque[i] = abs32(s.devmaxabs.Reduce(s.hDev.comp[i])) // do we need to / dt? some solvers use torque, some delta M...
	}
	minMz, maxMz := s.devmin.Reduce(s.mDev.comp[X]), s.devmax.Reduce(s.mDev.comp[X])
	maxtorque := max32(torque[0], max32(torque[1], torque[2]))

	s.tabwriter.Print(float32(s.time) * s.UnitTime())
	s.tabwriter.Print(m[Z], m[Y], m[X])
	s.tabwriter.Print(s.hextSI[Z], s.hextSI[Y], s.hextSI[X])
	s.tabwriter.Print(maxtorque)
	s.tabwriter.Print(minMz, maxMz)
	s.tabwriter.Print(s.autosaveIdx)
	s.tabwriter.Flush() // It's handy to have the output stored intermediately
	t.sinceoutput = float32(s.time) * s.UnitTime()
}

func m_average(m *tensor.T4) (mx, my, mz float32) {
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
	mx /= float32(count)
	my /= float32(count)
	mz /= float32(count)
	return
}


type MOmf struct {
	*Periodic
	format string
}

// INTERNAL
func (m *MOmf) Save(s *Sim) {
	fname := s.outputdir + "/" + "m" + fmt.Sprintf(FILENAME_FORMAT, s.autosaveIdx) + ".omf"
	var file omf.File
	file.T4 = s.mLocal
	file.StepSize = s.input.cellSize
	file.MeshUnit = "m"
	file.Desc = s.desc
	file.ValueMultiplier = s.mSat
	file.ValueUnit = "A/m"
	file.Format = m.format
	file.DataFormat = "4"
	omf.FEncode(fname, file)
	m.sinceoutput = float32(s.time) * s.UnitTime()
}


// INTERNAL
type Torque struct {
	*Periodic
	format string
}

func (m *Torque) Save(s *Sim) {
	fname := s.outputdir + "/" + "torque" + fmt.Sprintf(FILENAME_FORMAT, s.autosaveIdx) + ".omf"
	var file omf.File
	TensorCopyFrom(s.hDev, s.hLocal) //!
	file.T4 = s.hLocal
	file.StepSize = s.input.cellSize
	file.MeshUnit = "m"
	file.Desc = s.desc
	file.ValueMultiplier = 1
	file.ValueUnit = ""
	file.Format = m.format
	file.DataFormat = "4"
	omf.FEncode(fname, file)
	m.sinceoutput = float32(s.time) * s.UnitTime()
}


//Utility method for saving in .omf format.
func (s *Sim) saveOmf(data *tensor.T4, filename, unit, format string) {
	var file omf.File
	file.T4 = data
	file.StepSize = s.input.cellSize
	file.MeshUnit = "m"
	file.Desc = s.desc
	file.ValueMultiplier = 1
	file.ValueUnit = unit
	file.Format = format
	file.DataFormat = "4"
	omf.FEncode(filename, file)
}

//_________________________________________ png

// INTERNAL
type MPng struct {
	*Periodic
}

// INTERNAL
func (m *MPng) Save(s *Sim) {
	s.assureOutputUpToDate()
	fname := s.outputdir + "/" + "m" + fmt.Sprintf(FILENAME_FORMAT, s.autosaveIdx) + ".png"
	out := MustOpenWRONLY(fname)
	defer out.Close()
	draw.PNG(out, s.mLocal)
	m.sinceoutput = float32(s.time) * s.UnitTime()
}


//INTERNAL
func abs32(x float32) float32 {
	if x > 0 {
		return x
	}
	return -x
}

//INTERNAL
func max32(x, y float32) float32 {
	if x > y {
		return x
	}
	return y
}
