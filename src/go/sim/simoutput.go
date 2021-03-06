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
	"strings"
)

// reduce output resolution by N to save disk space.
var outputSubsample int = 1

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
	//var ovfver int
	//switch version {
	//default:
	//	panic("Unknown version of the ovf " + version + ". options are ovf1, ovf2. Falling back to default value of ovf2")  
	//	ovfver = 2
	//case "ovf1":
	//	ovfver = 1
	//case "ovf2":
	//	ovfver = 2
	//}

	switch what {
	default:
		panic("unknown output quantity " + what + ". options are: m, table")
	case "m":
		switch format {
		default:
			panic("unknown format " + format + ". options are: binary (=binary4,omf,binary4_2, omf_2), text (=ascii), png")
		case "png":
			return &MPng{&Periodic{0., 0.}}
		case "binary", "binary4", "omf":
			return &MOmf{&Periodic{0., 0.}, "binary", 1} // 1 tells the version of ovf format to be used
		case "binary_2", "binary4_2", "omf_2":
			return &MOmf{&Periodic{0., 0.}, "binary", 2} // 2 tells the version of ovf format to be used
		case "text", "ascii":
			return &MOmf{&Periodic{0., 0.}, "text", 1}
		}
	case "table":
		//format gets ignored for now
		return &Table{&Periodic{0., 0.}}
	case "phi", "energydensity":
		return &Edens{&Periodic{0., 0.}, format, 1}
	case "torque":
		return &Torque{&Periodic{0., 0.}, format, 1}
	}
	panic("bug")
	return nil // not reached
}

// How to format the id for the file name, e.g.: m000000001.omf
const FILENAME_FORMAT = "%08d"


type Table struct {
	*Periodic
}

// Index for s.input.tabulate 
// (which determines what output to write to the datatable)
const (
	TAB_TIME = iota
	TAB_M
	TAB_B
	TAB_J
	TAB_ID
	TAB_E
	TAB_MAXDMDT
	TAB_MINMAXMZ
	TAB_COREPOS
	TAB_LEN // Must be last in the list. Not used as key but to know the length of the tabulate array
)

var tabString []string = []string{"time", "m", "b", "j", "id", "e", "maxdm/dt", "minmaxmz", "corepos"}

func (s *Sim) initTabWriter() {
	if s.tabwriter != nil {
		panic(Bug("Tabwriter already initiated."))
	}
	fname := s.outputdir + "/" + "datatable.odt"
	out := MustOpenWRONLY(fname)
	Print("Opened " + fname)
	s.tabwriter = omf.NewTabWriter(out)
	if s.input.tabulate[TAB_TIME] {
		s.tabwriter.AddColumn("Time", "s")
	}
	if s.input.tabulate[TAB_M] {
		s.tabwriter.AddColumn("Mx/Ms", "")
		s.tabwriter.AddColumn("My/Ms", "")
		s.tabwriter.AddColumn("Mz/Ms", "")
	}
	if s.input.tabulate[TAB_B] {
		s.tabwriter.AddColumn("Bx", "T")
		s.tabwriter.AddColumn("By", "T")
		s.tabwriter.AddColumn("Bz", "T")
	}
	if s.input.tabulate[TAB_J] {
		s.tabwriter.AddColumn("jx", "A/m2")
		s.tabwriter.AddColumn("jy", "A/m2")
		s.tabwriter.AddColumn("jz", "A/m2")
	}
	if s.input.tabulate[TAB_E] {
		if IsInf(s.cellSize[X]) {
			s.tabwriter.AddColumn("Energy", "J/m")
		} else {
			s.tabwriter.AddColumn("Energy", "J")
		}
	}
	if s.input.tabulate[TAB_MAXDMDT] {
		s.tabwriter.AddColumn("max_dm/dt", "gammaMs")
	}
	if s.input.tabulate[TAB_MINMAXMZ] {
		s.tabwriter.AddColumn("min_Mz/Ms", "")
		s.tabwriter.AddColumn("max_Mz/Ms", "")
	}
	if s.input.tabulate[TAB_COREPOS] {
		s.tabwriter.AddColumn("core_x", "m")
		s.tabwriter.AddColumn("core_y", "m")
	}
	if s.input.tabulate[TAB_ID] {
		s.tabwriter.AddColumn("id", "")
	}
}

func (s *Sim) Tabulate(what string, tabulate bool) {
	if s.tabwriter != nil {
		panic(InputErr("tabulate must be called before the datatable is opened (Before run, desc, autosave, ...)."))
	}
	s.input.tabulate[indexof(strings.ToLower(what), tabString)] = tabulate
}

func indexof(key string, array []string) int {
	for i, v := range array {
		if v == key {
			return i
		}
	}
	panic(InputErr("Illegal argument: " + key + " options are: " + fmt.Sprint(array)))
}

func (t *Table) Save(s *Sim) {
	if s.tabwriter == nil {
		s.initTabWriter()
	}
	if s.input.tabulate[TAB_TIME] {
		s.tabwriter.Print(float32(s.time) * s.UnitTime())
	}
	if s.input.tabulate[TAB_M] {
		m := [3]float32{}
		for i := range m {
			m[i] = s.devsum.Reduce(s.mDev.comp[i]) / s.avgNorm
		}
		s.tabwriter.Print(m[Z], m[Y], m[X])
	}
	if s.input.tabulate[TAB_B] {
		s.tabwriter.Print(s.hextSI[Z], s.hextSI[Y], s.hextSI[X])
	}
	if s.input.tabulate[TAB_J] {
		s.tabwriter.Print(s.input.j[Z], s.input.j[Y], s.input.j[X])
	}
	if s.input.tabulate[TAB_E] {
		E := s.energy * s.UnitEnergy()
		//E := s.GetEnergySI()
		s.tabwriter.Print(E)
	}
	if s.input.tabulate[TAB_MAXDMDT] {
		//torque := [3]float32{}
		//for i := range torque {
		//	torque[i] = abs32(s.devmaxabs.Reduce(s.hDev.comp[i])) // do we need to / dt? some solvers use torque, some delta M...
		//}
		//maxtorque := max32(torque[0], max32(torque[1], torque[2]))
		s.tabwriter.Print(s.torque)
	}
	if s.input.tabulate[TAB_MINMAXMZ] {
		minMz, maxMz := s.devmin.Reduce(s.mDev.comp[X]), s.devmax.Reduce(s.mDev.comp[X])
		s.tabwriter.Print(minMz, maxMz)
	}
	if s.input.tabulate[TAB_COREPOS] {
		corepos := s.corePos()
		s.tabwriter.Print(corepos[0], corepos[1])
	}
	if s.input.tabulate[TAB_ID] {
		s.tabwriter.Print(s.autosaveIdx)
	}
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
	ovfversion int
}

// INTERNAL
func (m *MOmf) Save(s *Sim) {
	fname := s.outputdir + "/" + "m" + fmt.Sprintf(FILENAME_FORMAT, s.autosaveIdx) + ".omf"
	var file omf.File
	file.T4 = subsampleOutput(s.mLocal)
	file.StepSize = s.input.cellSize
	for i := range file.StepSize {
		file.StepSize[i] *= float32(outputSubsample)
	}
	file.MeshUnit = "m"
	file.Desc = s.desc
	file.ValueMultiplier = s.mSat
	file.ValueUnit = "A/m"
	file.Format = m.format
	file.OVFVersion = m.ovfversion
	file.DataFormat = "4"
	file.StageTime = float64(m.Periodic.period)
	file.TotalTime = s.time * float64(s.UnitTime())
	omf.FEncode(fname, file)
	m.sinceoutput = float32(s.time) * s.UnitTime()
}

var subsampleBuffer *tensor.T4

func subsampleOutput(in *tensor.T4) *tensor.T4 {
	if outputSubsample == 1 {
		return in
	}

	size2 := []int{3, 0, 0, 0}
	copy(size2, in.Size())
	for i := 1; i < 4; i++ {
		size2[i] /= outputSubsample
		if size2[i] == 0 {
			size2[i] = 1
		}
	}
	if subsampleBuffer == nil {
		subsampleBuffer = tensor.NewT4(size2)
	}
	if !tensor.EqualSize(subsampleBuffer.Size(), size2) {
		subsampleBuffer = tensor.NewT4(size2)
	}
	subsample4(in, subsampleBuffer, outputSubsample)
	return subsampleBuffer
}


type Edens struct {
	*Periodic
	format string
	ovfversion int
}


var edens4 *tensor.T4
var edens3 *tensor.T3

// INTERNAL
func (m *Edens) Save(s *Sim) {
	s.initEDens()
	// no initEDens() to make sure it has been calculated already
	if edens4 == nil {
		edens4 = tensor.NewT4(s.size4D[:])
		edens3 = tensor.ToT3(tensor.Component(edens4, 0))
	}
	TensorCopyFrom(s.phiDev, edens3)

	fname := s.outputdir + "/" + "phi" + fmt.Sprintf(FILENAME_FORMAT, s.autosaveIdx) + ".omf"
	var file omf.File
	file.T4 = subsampleOutput(edens4)
	file.StepSize = s.input.cellSize
	file.MeshUnit = "m"
	file.Desc = s.desc
	file.ValueMultiplier = s.mSat
	file.ValueUnit = "internal"
	file.Format = m.format
	file.OVFVersion = m.ovfversion
	file.StageTime = float64(m.Periodic.period)
	file.TotalTime = s.time * float64(s.UnitTime())
	file.DataFormat = "4"
	omf.FEncode(fname, file)
	m.sinceoutput = float32(s.time) * s.UnitTime()
}


// INTERNAL
type Torque struct {
	*Periodic
	format string
	ovfversion int
}

func (m *Torque) Save(s *Sim) {
	fname := s.outputdir + "/" + "torque" + fmt.Sprintf(FILENAME_FORMAT, s.autosaveIdx) + ".omf"
	var file omf.File
	TensorCopyFrom(s.hDev, s.hLocal) //!
	file.T4 = subsampleOutput(s.hLocal)
	file.StepSize = s.input.cellSize
	file.MeshUnit = "m"
	file.Desc = s.desc
	file.ValueMultiplier = 1
	file.ValueUnit = ""
	file.Format = m.format
	file.OVFVersion = m.ovfversion
	file.StageTime = float64(m.Periodic.period)
	file.TotalTime = s.time * float64(s.UnitTime())
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
	file.OVFVersion = 1
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
	draw.PNG(out, subsampleOutput(s.mLocal))
	m.sinceoutput = float32(s.time) * s.UnitTime()
}


func (s *Sim) SubsampleOutput(factor int) {
	outputSubsample = factor
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
