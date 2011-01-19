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
	"draw"
	"iotool"
	"omf"
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
		panic("unknown output quantity " + what + ". options are: m, torque, table")
	case "m":
		switch format {
		default:
			panic("unknown format " + format + ". options are: binary, ascii, png")
			//		case "binary":
			//			return &MBinary{&Periodic{0., 0.}}
			//		case "ascii":
			//			return &MAscii{&Periodic{0., 0.}}
		case "png":
			return &MPng{&Periodic{0., 0.}}
		case "omf":
			return &MOmf{&Periodic{0., 0.}}
		}
		//	case "torque":
		//		switch format {
		//		default:
		//			panic("unknown format " + format + ". options are: binary, ascii, png")
		//		case "binary":
		//			return &TorqueBinary{&Periodic{0., 0.}}
		//		}

	case "table":
		//format gets ignored for now
		return &Table{&Periodic{0., 0.}}
	}

	panic("bug")
	return nil // not reached
}

//__________________________________________ ascii

// Opens a file for writing 
//func fopen(filename string) *os.File {
//	file, err := os.Open(filename, os.O_WRONLY|os.O_CREAT, 0666)
//	if err != nil {
//		panic(err)
//	}
//	return file
//}


// TODO: it would be nice to have a separate date sturcture for the format and one for the data.
// with a better input file parser we could allow any tensor to be stored:
// save average(component(m, z)) jpg
// INTERNAL
//type MAscii struct {
//	*Periodic
//}

const FILENAME_FORMAT = "%08d"

// INTERNAL
//func (m *MAscii) Save(s *Sim) {
//	fname := s.outputdir + "/" + "m" + fmt.Sprintf(FILENAME_FORMAT, s.autosaveIdx) + ".tensor"
//	file := fopen(fname)
//	defer file.Close()
//	tensor.WriteMetaTensorAscii(file, s.mLocal, s.desc)
//	m.sinceoutput = float32(s.time) * s.UnitTime()
//}

type Table struct {
	*Periodic
}

func (t *Table) Save(s *Sim) {
	if s.tabwriter == nil {
		fname := s.outputdir + "/" + "datatable.odt"
		out := iotool.MustOpenWRONLY(fname)
		s.tabwriter = omf.NewTabWriter(out)
		s.tabwriter.AddColumn("Time", "s")
		s.tabwriter.AddColumn("Mx/Ms", "")
		s.tabwriter.AddColumn("My/Ms", "")
		s.tabwriter.AddColumn("Mz/Ms", "")
		s.tabwriter.AddColumn("Bx", "T")
		s.tabwriter.AddColumn("By", "T")
		s.tabwriter.AddColumn("Bz", "T")
		s.tabwriter.AddColumn("id", "")
	}

	// calculate reduced quantities
	m := [3]float32{}
	torque := [3]float32{}
	N := Len(s.size3D)
	for i := range m {
		m[i] = s.devsum.Reduce(s.mDev.comp[i]) / float32(N)
		torque[i] = abs32(s.devmaxabs.Reduce(s.hDev.comp[i]) / s.dt)
	}
	//minMz, maxMz := s.devmin.Reduce(s.mDev.comp[X]), s.devmax.Reduce(s.mDev.comp[X])

	s.tabwriter.Print(float32(s.time) * s.UnitTime())
	s.tabwriter.Print(m[Z], m[Y], m[X])
	s.tabwriter.Print(s.hextSI[Z], s.hextSI[Y], s.hextSI[X])

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

//_________________________________________ binary

// INTERNAL
//type MBinary struct {
//	*Periodic
//}
//
// INTERNAL
// TODO: files are not closed?
// TODO/ also for writeAscii
//func (m *MBinary) Save(s *Sim) {
//	fname := s.outputdir + "/" + "m" + fmt.Sprintf(FILENAME_FORMAT, s.autosaveIdx) + ".tensor"
//	out := fopen(fname)
//	defer out.Close()
//	tensor.WriteMetaTensorBinary(out, s.mLocal, s.desc)
//	m.sinceoutput = float32(s.time) * s.UnitTime()
//}


type MOmf struct {
	*Periodic
}

// INTERNAL
// TODO: format: text
func (m *MOmf) Save(s *Sim) {
	fname := s.outputdir + "/" + "m" + fmt.Sprintf(FILENAME_FORMAT, s.autosaveIdx) + ".omf"
	var file omf.File
	file.T4 = s.mLocal
	file.StepSize = s.input.cellSize
	file.MeshUnit = "m"
	file.Desc = s.desc
	file.ValueMultiplier = s.mSat
	file.ValueUnit = "A/m"
	file.Format = "binary"
	file.DataFormat = "4"
	omf.FEncode(fname, file)
	m.sinceoutput = float32(s.time) * s.UnitTime()
}


// INTERNAL
type TorqueBinary struct {
	*Periodic
}

// TODO: quick and dirty for the moment
//func (m *TorqueBinary) Save(s *Sim) {
//	fname := s.outputdir + "/" + "torque" + fmt.Sprintf(FILENAME_FORMAT, s.autosaveIdx) + ".tensor"
//	out := fopen(fname)
//	defer out.Close()
//	TensorCopyFrom(s.hDev, s.hLocal) //!
//	tensor.WriteMetaTensorBinary(out, s.hLocal, s.desc)
//	m.sinceoutput = float32(s.time) * s.UnitTime()
//}


//_________________________________________ png

// INTERNAL
type MPng struct {
	*Periodic
}

// INTERNAL
func (m *MPng) Save(s *Sim) {
	fname := s.outputdir + "/" + "m" + fmt.Sprintf(FILENAME_FORMAT, s.autosaveIdx) + ".png"
	out := iotool.MustOpenWRONLY(fname)
	defer out.Close()
	draw.PNG(out, s.mLocal)
	m.sinceoutput = float32(s.time) * s.UnitTime()
}

// To implement omf.Interface:
// TODO: generalize
//func (s *Sim) GetData() (data tensor.Interface, multiplier float32, unit string) {
//	return s.mLocal, s.input.msat, "A/m"
//}
//
//func (s *Sim) GetMesh() (cellsize []float32, unit string) {
//	return s.input.cellSize[:], "m"
//}
//
//func (s *Sim) GetMetadata() map[string]string {
//	return s.metadata
//}


//INTERNAL
func abs32(x float32) float32 {
	if x > 0 {
		return x
	}
	return -x
}
