package sim

import ()

// Field contains everything neccesary to calculate the effective field H_eff
type Field struct {
	Magnet

	*Conv
	Hext []float

	// Exchange
	// Anis
	// External
}

// Takes the parameters from a Magnet struct and
// initializes everything.
func NewField(dev *Backend, mag *Magnet, demag_accuracy int) *Field { // todo: do not need backend param here
	field := new(Field)

	field.Magnet = *mag
	field.Hext = nil
	//field.Start("Kernel calculation")
	Debugv("Calculating kernel")
	demag := FaceKernel6(field.paddedsize, field.cellSize, demag_accuracy)
	exch := Exch6NgbrKernel(field.paddedsize, field.cellSize)
	// Add Exchange kernel to demag kernel
	for i := range demag {
		D := demag[i].List()
		E := exch[i].List()
		for j := range D {
			D[j] += E[j]
		}
	}
	//field.Stop("Kernel calculation")
	field.Conv = NewConv(dev, field.size, demag)

	return field
}


// Calculates the effective field of m and stores it in h
func (f *Field) CalcHeff(m, h *Tensor) {
	Debugvv("Field.CalcHeff()")
	f.Convolve(m, h)

	if f.Hext != nil {
		for i := range f.hComp {
			f.AddConstant(f.hComp[i], f.Hext[i])
		}
	}
}


func (f *Field) String() string {
	return "Field:\n" + f.Magnet.String()
}
