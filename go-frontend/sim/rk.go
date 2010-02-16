package sim

// import(
//   . "../tensor";
//   . "math";
// )
// 
// 
// // "constants"
// 
// var NAN = float(NaN());
// 
// // RKF
// 
// var butcher45 = [...][5]float{
//   [5]float {   NAN,         NAN,           NAN,           NAN,           NAN},
//   [5]float {   1.0/4.0,     NAN,           NAN,           NAN,           NAN},
//   [5]float {   3.0/32.0,    9.0/32.0,      NAN,           NAN,           NAN},
//   [5]float {1932.0/2197.0, -7200.0/2197.0, 7296.0/2197.0, NAN,           NAN},
//   [5]float { 439.0/216.0,  -8.0,           3680.0/513.0, -845.0/4104.0,  NAN},
//   [5]float {  -8.0/27.0,    2.0,          -3544.0/2565.0, 1859.0/4104.0, -11.0/40.0}};
// 
// var       h45 = [6]float{NAN, 1.0/4.0, 3.0/8.0, 12.0/13.0, 1.0, 1.0/2.0};
// var weights45 = [6]float{16.0/135.0, 0.0, 6656.0/12825.0, 28561.0/56430.0, -9.0/50.0, 2.0/55.0};
// var weights54 = [6]float{25.0/216.0, 0.0, 1408.0/2565.0,  2197.0/4104.0,   -1.0/5.0,  0.0};
// 
// // RK4
// 
// var butcher4 = [...][4]float{
//   [4]float{NAN, NAN, NAN, NAN},
//   [4]float{1.0 / 2.0, NAN, NAN},
//   [4]float{0.0, 1.0 / 2.0, NAN},
//   [4]float{0.0, 0.0, 1.0, NAN}};
// 
// var      h4 = [...]float{NAN, 1.0 / 2.0, 1.0 / 2.0, 1.0};
// var weight4 = [...]float{1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0};
// 
// /////
// 
// 
// type RK4 struct{
//   h *Tensor4;
//   m, m0 *Tensor4;
//   k [4]*Tensor4;
//   
//   field *FieldPlan;
//   t, dt float;
// }
// 
// func NewRK4(m *Tensor4, field *FieldPlan, dt float){
//   rk = new(RK4);
//   rk.m = m;
//   rk.field = field;
//   rk.dt = dt;
// 
//   rk.m0 = NewTensor4(m.Size());
//   rk.k = [4]*Tensor4;
//   for i:=range(rk.k){
//     rk.k[i] = NewTensor4(m.Size());
//   }
// }
// 
// func (rk *RK4) step(){
//   M := rk.m;
//   Mx := M.Component(X);
//   My := M.Component(Y);
//   Mz := M.Component(Z);
//   M0 := rk.m0;
//   H := rk.h;
//   Hx := H.Component(X);
//   Hy := H.Component(Y);
//   Hz := H.Component(Z);
// 
//   m := []float{0., 0., 0.};
//   h := []float{0., 0., 0.};
//   torque := []float{0., 0., 0.};
// 
//   M0.SetTo(M);
//   for i:=range(M.Component(X)){
//     m[X] = Mx[i];
//     m[Y] = My[i];
//     m[Z] = Mz[i];
//     
// 
//   }
//   
// //     //t0 := rk.t;
// //     alpha := 1.0;
// //     gilbert := 1.0 / (1.0 + alpha * alpha);
// //     m, h := make([]float, 3), make([]float, 3);
// // 
// //     //CopyTo(rk.m, rk.m0);
// //     rk.field.Execute(rk.m, rk.h);
// // 
// //     for i:=range(rk.m.Component(0).List()){
// //        m[0] = rk.m.Component(0).List()[i];
// //        m[1] = rk.m.Component(1).List()[i];
// //        m[2] = rk.m.Component(2).List()[i];
// //        h[0] = rk.h.Component(0).List()[i];
// //        h[1] = rk.h.Component(1).List()[i];
// //        h[2] = rk.h.Component(2).List()[i];
// //        Torque(m, h, rk.k.Array()[i], alpha, gilbert); 
// //     }
// // 
// // 
// //     //butcher tableau
// //     for i:=range(weight) {
// // 
// //       m[0] = rk.m.Component(0).List()[i];
// //       m[1] = rk.m.Component(1).List()[i];
// //       m[2] = rk.m.Component(2).List()[i];
// // 
// //       //set time and update
// //       //final double[] butcherI = butcher[i];
// //       for c:=range(rk.m.Component(0).List()){
// //             //rkc = rk[c];
// //             //reset m
// //             //cell.m.set(rk[c].m0); inlined:
// //             //k = rk[c].k;
// //             for j:=0; j < i; j++ {
// //               //cell.m.add(dt * butcher[i][j], k[i-1]); inlined:
// //               rk.m.Component(0).List()[i] += rk.k.Array()[i][0] * rk.dt;
// // 	      rk.m.Component(1).List()[i] += rk.k.Array()[i][1] * rk.dt;
// // 	      rk.m.Component(2).List()[i] += rk.k.Array()[i][2] * rk.dt;
// // 	      for 
// //             }
// // 
// //             m.normalize();
// //             // push_down;
// //             // cell.child1.m.set(m);
// //             // cell.child2.m.set(m);
// //             // it seems the java function call overhead trashes performance if we
// //             // propagate m down recursively, so let's hand-code it ... sigh.
// //             // so we have to limit ourselves to a few levels, should be ok most
// //             // of the time but we should check this once...
// //             // alas, this cripples the output, unles we only use 2 adaptive mesh levels.
// //             final Cell child1 = cell.child1;
// //             final Cell child2 = cell.child2;
// //             if (child1 != null) {
// //               child1.m.x = child2.m.x = m.x;
// //               child1.m.y = child2.m.y = m.y;
// //               child1.m.z = child2.m.z = m.z;
// // 
// //               if (child1.child1 != null) {
// //                 child1.child1.m.x = child1.child2.m.x = child2.child1.m.x = child2.child2.m.x;
// //                 child1.child1.m.y = child1.child2.m.y = child2.child1.m.y = child2.child2.m.y;
// //                 child1.child1.m.z = child1.child2.m.z = child2.child1.m.z = child2.child2.m.z;
// //               }
// //             }
// //           }
// //           c++;
// //         }
// //       }
// //       sim.totalTime = t0 + h[i] * dt;
// //       sim.update();
// // 
// //       //new k
// //       {
// //         int c = 0;
// //         for (Cell cell = sim.mesh.coarseRoot; cell != null; cell = cell.next) {
// //           if (cell.updateLeaf) {
// //             //torque(cell.m, cell.h, rk[c].k[i]);
// //             rk[c].k[i] = cell.dmdt;
// //           }
// //           c++;
// //         }
// //       }
// //     }
// // 
// // 
// //     //new m
// //     {
// //       int c = 0;
// //       for (Cell cell = sim.mesh.coarseRoot; cell != null; cell = cell.next) {
// //         if (cell.updateLeaf) {
// //           //reset
// // 
// //           final RKData rkc = rk[c];
// //           final Vector m = cell.m;
// // 
// //           //cell.m.set(rk[c].m0); inlined:
// //           m.x = rkc.m0.x;
// //           m.y = rkc.m0.y;
// //           m.z = rkc.m0.z;
// // 
// //           Vector[] k = rk[c].k;
// // 
// //           for (int i = 0; i < weight.length; i++) {
// //             m.add(dt * weight[i], k[i]);
// //           }
// // 
// //           m.normalize();
// //           // push_down.
// // 
// //           final Cell child1 = cell.child1;
// //             final Cell child2 = cell.child2;
// //             if (child1 != null) {
// //               child1.m.x = child2.m.x = m.x;
// //               child1.m.y = child2.m.y = m.y;
// //               child1.m.z = child2.m.z = m.z;
// // 
// //               if (child1.child1 != null) {
// //                 child1.child1.m.x = child1.child2.m.x = child2.child1.m.x = child2.child2.m.x;
// //                 child1.child1.m.y = child1.child2.m.y = child2.child1.m.y = child2.child2.m.y;
// //                 child1.child1.m.z = child1.child2.m.z = child2.child1.m.z = child2.child2.m.z;
// //               }
// //             }
// // 
// //         }
// //         c++;
// //       }
// //     }
// //     sim.update();
// // 
// //     // (3) bookkeeping
// //     sim.totalTime = t0 + dt;
// //     totalSteps++;*/
// }
