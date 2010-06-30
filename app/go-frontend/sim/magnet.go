package sim

import(
  "fmt"
  "io"
)


type Magnet struct{

  Material
  size []int;   ///< Mesh Size, e.g. 4x64x64
  cellSize []float;  ///< Cell Size in exchange lengths, e.g. 0.5 x 0.5 x 1.2

}

func(mag Magnet) Size() []int {
  return mag.size
}


func(mag Magnet) CellSize() []float {
  return mag.cellSize
}


func(mag Magnet) PrintGeom(out io.Writer){   ///@todo String()
  //fmt.Fprintln(out, "Geometry");
  fmt.Fprintln(out, "Grid Size  : \t", mag.size);
  fmt.Fprint(out, "Cell Size  : \t");
  for i:=range(mag.cellSize){
    fmt.Fprint(out, mag.UnitLength() * mag.cellSize[i], " ");
  }
  fmt.Fprint(out, "(m), (");
   for i:=range(mag.cellSize){
    fmt.Fprint(out, mag.cellSize[i], " ");
  }
  fmt.Fprintln(out, "exch. lengths)");

  fmt.Fprint(out, "Sim Size   : \t ");
  for i:=range(mag.size){
    fmt.Fprint(out, float(mag.size[i]) * mag.UnitLength() * mag.cellSize[i], " ");
  }
  fmt.Fprintln(out, "(m)");
}


