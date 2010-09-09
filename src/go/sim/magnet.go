package sim

import (
	"fmt"
)

// represents the thing being simulated
type Magnet struct {

}


func NewMagnet(dev *Backend, mat *Material, size []int, cellSize []float) *Magnet {
	m, h := NewTensor(dev, Size4D(size)), NewTensor(dev, Size4D(size))
	mComp, hComp := [3]*DevTensor{}, [3]*DevTensor{}
	for i := range mComp {
		mComp[i] = m.Component(i)
		hComp[i] = h.Component(i)
	}
	paddedsize := padSize(size)
	return &Magnet{*mat, size, paddedsize, cellSize, m, h, mComp, hComp}
}


// func (mag *Magnet) Size() []int {
// 	return mag.size
// }

func (mag *Magnet) Size4D() []int {
	return Size4D(mag.size)
}

func (mag *Magnet) NSpins() int {
	return mag.size[X] * mag.size[Y] * mag.size[Z]
}

// func (mag *Magnet) CellSize() []float {
// 	return mag.cellSize
// }

func (mag *Magnet) M() *DevTensor {
	return mag.m
}


func (mag *Magnet) String() string {
	s := "Magnet:\n"
	s += mag.Material.String()
	s += fmt.Sprintln("Grid Size  : \t", mag.size)
	s += fmt.Sprint("Cell Size  : \t")
	for i := range mag.cellSize {
		s += fmt.Sprint(mag.UnitLength()*mag.cellSize[i], " ")
	}
	s += fmt.Sprint("(m), (")
	for i := range mag.cellSize {
		s += fmt.Sprint(mag.cellSize[i], " ")
	}
	s += fmt.Sprintln("exch. lengths)")

	s += fmt.Sprint("Sim Size   : \t ")
	for i := range mag.size {
		s += fmt.Sprint(float(mag.size[i])*mag.UnitLength()*mag.cellSize[i], " ")
	}
	s += fmt.Sprintln("(m)")
	return s
}
