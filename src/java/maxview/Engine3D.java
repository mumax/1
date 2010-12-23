import javax.swing.*;
import java.awt.*;

public class Engine3D {

    public static void main (String args[]) {
		Group root = new Group();
	
		root.add(SchuifModels.bord(6,6));		
		Brush stone = SchuifModels.stone(0);
		stone.translate(-1, 0, 1);
		root.add(stone);
		stone = SchuifModels.stone(1);
		root.add(stone);
		
		Universe universe = new Universe(Color.WHITE, new Vertex(0, 5, 0), 1);
		View view = new View(universe);
		view.setBord(10, 0, 0.3);
		universe.setRoot(root);
		
		JFrame frame = new JFrame("3D Engine");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setContentPane(view);
		frame.setSize(400, 300);
		frame.show();
	}
}
