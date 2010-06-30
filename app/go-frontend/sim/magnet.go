package sim

import(
  "fmt"
  "io"
)


type Magnet struct{

  Material
  Size []int;   ///< Mesh Size, e.g. 4x64x64
  CellSize []float;  ///< Cell Size in exchange lengths, e.g. 0.5 x 0.5 x 1.2

}


func(mag Magnet) PrintGeom(out io.Writer){   ///@todo String()
  //fmt.Fprintln(out, "Geometry");
  fmt.Fprintln(out, "Grid Size  : \t", mag.Size);
  fmt.Fprint(out, "Cell Size  : \t");
  for i:=range(mag.CellSize){
    fmt.Fprint(out, mag.UnitLength() * mag.CellSize[i], " ");
  }
  fmt.Fprint(out, "(m), (");
   for i:=range(mag.CellSize){
    fmt.Fprint(out, mag.CellSize[i], " ");
  }
  fmt.Fprintln(out, "exch. lengths)");

  fmt.Fprint(out, "Sim Size   : \t ");
  for i:=range(mag.Size){
    fmt.Fprint(out, float(mag.Size[i]) * mag.UnitLength() * mag.CellSize[i], " ");
  }
  fmt.Fprintln(out, "(m)");
}


