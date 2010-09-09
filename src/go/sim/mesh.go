package sim

type Mesh struct{
  size         []int      // Mesh Size, e.g. 4x64x64 TODO get rid of: already in FFT
  paddedsize   []int      // Mesh size with zero padding.
  cellSize     []float    // Cell Size in exchange lengths, e.g. Inf x 0.5 x 0.5
}