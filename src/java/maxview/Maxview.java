/**
  This file is part of a 3D engine,
  copyright Arne Vansteenkiste 2006-2010.
  Use of this source code is governed by the GNU General Public License version 3,
  as published by the Free Software Foundation.
*/
import javax.swing.*;
import java.awt.*;
import java.io.*;

public class Maxview {

    public static void main (String args[]) throws IOException{

      readFile(args[0]);

     Group root = new Group();

     Brush cone = Factory.cone(0.5, 32, 1);
     cone.setFillColor(Color.RED);
      root.add(cone);

    Universe universe = new Universe(Color.WHITE, new Vertex(2, 5, 0), 1);
    View view = new View(universe);
    view.setBord(10, 0, 0.3);
    universe.setRoot(root);

    JFrame frame = new JFrame("3D Engine");
    frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    frame.setContentPane(view);
    frame.setSize(400, 300);
    frame.show();
	}

  private static void readFile(String file) throws IOException{
    StreamTokenizer in = new StreamTokenizer(new InputStreamReader(new FileInputStream(new File(file))));

    int tok = in.nextToken();
    while(tok != StreamTokenizer.TT_EOF){
      System.out.println("" + in.nval);
      tok = in.nextToken();
    }
  }
}
