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
import static java.lang.Math.abs;
import refsh.*;
import java.awt.Color;
import java.awt.Graphics2D;
import java.io.IOException;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;

public class Maxview {

  Group root;
  Universe universe;
  View view;
  int width, height;
  int detail;

  public Maxview(){
    root = new Group();
    universe = new Universe(Color.WHITE, new Vertex(2, 5, 20), 0.8);
    view = new View(universe);
    view.setBord(25, 0, 0);
    universe.setRoot(root);
    width = height = 512;
  }


  public void exit(){
    System.exit(0);
  }

  /** Shows an interactive window. */
  public void show(){
    updateLight();
    JFrame frame = new JFrame("Maxview");
    frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
    frame.setContentPane(view);
    frame.setSize(width, height);
    frame.show();
  }

  /** Saves picture as png. */
  public void save(String filename) throws IOException{
        updateLight();
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_4BYTE_ABGR);
        Graphics2D graphics = (Graphics2D)(img.getGraphics());
        graphics.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
        graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        view.paint(graphics, width, height);
        ImageIO.write(img, "png", new File(filename));
//         System.out.println("saved " + filename);
  }

  
  public void updateLight(){
    root.light(universe);
  }


  /** Puts an arrow at position x,y,z, pointing in direction mx,my,mz */
  public void vec(double x, double y, double z, double mx, double my, double mz){
    Brush cone = Factory.cone(0.35, detail, 0.8);
      
      cone.rotate(0, -Math.PI/2);
      cone.rotate(-Math.PI/2, 0);

      double r = mx*mx + my*my + mz*mz;
      mx /= r;
      my /= r;
      mz /= r;
      double theta = Math.atan2(my, mx);
      double phi = -Math.asin(mz);

      cone.rotate(phi, 0);
      cone.rotateZ(theta);

      cone.setFillColor(ColorMap3D.map(mx, my, mz));
      cone.translate(-x, y, z);
      root.add(cone);

      // maximum coordinate for autozoom
      // (+.5) so that the cones of size 1 would fit entirely in the picture
      if( abs(x)+.5 > view.maxX) { view.maxX = abs(x)+.5; }
      if( abs(y)+.5 > view.maxY) { view.maxY = abs(y)+.5; }
  }


  /** Set canvas size of output, in pixels */
  public void size(int width, int height){
    this.width = width;
    this.height = height;
  }

  /** Set cone detail (nubmer of vertices) */
  public void detail(int vertices){
    this.detail = vertices;
  }
  
  public static void main (String args[]) throws Exception{
     Maxview maxview = new Maxview();
     Interpreter sh = new Interpreter(maxview.getClass(), maxview);
     new RefSh(sh).readstdin();
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
//     }

}
