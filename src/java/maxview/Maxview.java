/**
  This file is part of a 3D engine,
  copyright Arne Vansteenkiste 2006-2010.
  Use of this source code is governed by the GNU General Public License version 3,
  as published by the Free Software Foundation.
*/
package maxview;
import javax.swing.*;
import java.awt.*;
import java.io.*;
import java.lang.Math.*;
import refsh.*;

public class Maxview {

  Group root;
  Universe universe;
  View view;

  public Maxview(){
    root = new Group();
    universe = new Universe(Color.WHITE, new Vertex(2, 5, 20), 0.4);
    view = new View(universe);
    view.setBord(25, 0, 0);
    universe.setRoot(root);
  }

  public void exit(){
    System.exit(0);
  }

  public static void main (String args[]) throws IOException{

     Maxview maxview = new Maxview();
     Interpreter sh = new Interpreter(maxview.getClass(), maxview);
     new RefSh(sh).interactive();
    /*
    JFrame frame = new JFrame("Maxview");
    frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    frame.setContentPane(view);
    frame.setSize(800, 800);
    frame.show();*/
  }

//   private static void readStdin(Group root) throws IOException{
//     StreamTokenizer in = new StreamTokenizer(new InputStreamReader(System.in));
// 
//     int tok = in.nextToken();
//     while(tok != StreamTokenizer.TT_EOF){
//       double x = in.nval; tok = in.nextToken();
//       double y = in.nval; tok = in.nextToken();
//       double z = in.nval; tok = in.nextToken();
//       double mx = in.nval; tok = in.nextToken();
//       double my = in.nval; tok = in.nextToken();
//       double mz = in.nval; tok = in.nextToken();
// 
// 
//       Brush cone = Factory.cone(0.35, 32, 0.8);
//       
//       cone.rotate(0, -Math.PI/2);
//       cone.rotate(-Math.PI/2, 0);
// 
//       double r = mx*mx + my*my + mz*mz;
//       mx /= r;
//       my /= r;
//       mz /= r;
//       double theta = Math.atan2(my, mx);
//       double phi = -Math.asin(mz);
// 
//       cone.rotate(phi, 0);
//       cone.rotateZ(theta);
// 
//       cone.setFillColor(ColorMap3D.map(mx, my, mz));
//       cone.translate(-x, y, z);
//       root.add(cone);
// 
//       
//     }

}
