package sim

import(
  "fmt"
)


type Magnet struct{

  Material
  size []int        // Mesh Size, e.g. 4x64x64
  cellSize []float  // Cell Size in exchange lengths, e.g. 0.5 x 0.5 x 1.2

}


func NewMagnet(mat *Material, size []int, cellSize []float) *Magnet{
  return &Magnet{*mat, size, cellSize}
}


func(mag *Magnet) Size() []int {
  return mag.size
}

func(mag *Magnet) NSpins() int{
  return mag.size[X] * mag.size[Y] * mag.size[Z]
}


func(mag *Magnet) CellSize() []float {
  return mag.cellSize
}


func(mag *Magnet) String() string{
  s := "Magnet:\n"
  s += mag.Material.String()
  s += fmt.Sprintln("Grid Size  : \t", mag.size);
  s += fmt.Sprint("Cell Size  : \t");
  for i:=range(mag.cellSize){
    s += fmt.Sprint(mag.UnitLength() * mag.cellSize[i], " ");
  }
  s += fmt.Sprint("(m), (");
   for i:=range(mag.cellSize){
    s += fmt.Sprint(mag.cellSize[i], " ");
  }
  s += fmt.Sprintln("exch. lengths)");

  s += fmt.Sprint("Sim Size   : \t ");
  for i:=range(mag.size){
    s += fmt.Sprint(float(mag.size[i]) * mag.UnitLength() * mag.cellSize[i], " ");
  }
  s += fmt.Sprintln("(m)");
  return s
}
