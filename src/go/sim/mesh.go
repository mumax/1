package sim

type Mesh struct {
	size       [3]int   // Mesh Size, e.g. 4x64x64 TODO get rid of: already in FFT
	size4D     [4]int   // Vector-field-size of, a.o., the magnetization. 3 x N0 x N1 x N2
	paddedsize []int   // Mesh size with zero padding.
	cellSize   [3]float // Cell Size in exchange lengths, e.g. Inf x 0.5 x 0.5
}
