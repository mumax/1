/**
  This file is part of a 3D engine,
  copyright Arne Vansteenkiste 2006-2010.
  Use of this source code is governed by the GNU General Public License version 3,
  as published by the Free Software Foundation.
*/
import java.awt.Color;

public final class SchuifModels {

	public static final Color[] BACKGROUND = new Color[]{
		new Color(0, 0, 130), new Color(255, 255, 255)},
	PLAYER = new Color[]{new Color(255, 0, 0), new Color(255, 255, 0)};
	
	public static Group bord(int width, int height){
		Group bord = new Group(false);
		for(int i = 0; i < width; i++){
			for(int j = 0; j < height; j++){
				Mesh sheet = Factory.sheet(1, 1);
				sheet.translate(i, 0, j);
				sheet.setFillColor(BACKGROUND[(i+j)%2]);
				bord.add(sheet);
			}
		}
		bord.translate(-((double)width-1)/2.0, 0, -((double)height-1)/2.0);
		return bord;
	}
	
	public static Brush stone(int player){
		Mesh stone = Factory.cylinder(0.42, 32, 0.2);
		stone.setFillColor(PLAYER[player]);
		return stone;
	}
}
