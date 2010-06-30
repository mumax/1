package sim

import(
  "tensor"
)

/// Fills the tensor with a uniform magnetization
/// @todo get rid of Tensor4
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
