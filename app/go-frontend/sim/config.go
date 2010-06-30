package sim

import(
  "tensor"
)

func Uniform(m *tensor.Tensor4, x, y, z float){
  a := m.Array();
  for i:=range(a[0]){
    for j:=range(a[0][i]){
      for k:=range(a[0][i][j]){
	a[X][i][j][k] = x;
	a[Y][i][j][k] = y;
	a[Z][i][j][k] = z;
      }
    }
  }
}
