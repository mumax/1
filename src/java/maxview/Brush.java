/**
  This file is part of a 3D engine,
  copyright Arne Vansteenkiste 2006-2010.
  Use of this source code is governed by the GNU General Public License version 3,
  as published by the Free Software Foundation.
*/
import java.io.Serializable;
import java.awt.Color;
import static java.lang.Math.sin;
import static java.lang.Math.cos;

/**
Object dat weergegeven kan worden in een 3D Universe.
*/
public abstract class Brush implements Serializable{
	
	/** Hoekpunten van het Object, referenties worden gedeeld met eventuele sub-Objecten. */
	public abstract Vertex[] getVertices();
	
	/** Herbereken de kleuren, gebruik makend van de belichting in Universe u. */
	public abstract void light(Universe u);
	
	/** Getal voor ordening van de 3D Objecten: verste z-coordinaat. */
	public abstract double getZ();
	
	/** Tekent de Brush door de ogen van View v */ 
	public abstract void paint(View v);
	
	/** Maakt copie. */
	public abstract Brush copy();
	
	/** Sorteert eventuele sub-Brushes in volgorde van tekenen. */
	public abstract void sort();
	
	public abstract void setFillColor(Color c);
	
	public abstract void setLineColor(Color c);
	
	/** Verschuiving werkt op alle vertices. */
	public void translate(double dx, double dy, double dz){
		Vertex[] vertex = getVertices();
		for(int i = 0; i < vertex.length; i++)
			vertex[i].translate(dx, dy, dz);
	}

  public void rotate(double theta, double phi){
    rotateVertexArray(getVertices(), theta, phi);
  }

public static void rotateVertexArray(Vertex[] v, double phi, double theta){

       for(int i=0; i<v.length; i++){
          Vertex V = v[i];


          double m11 = cos(phi);
          double m12 = 0;
          double m13 = -sin(phi);

          double m21 = -sin(phi)*sin(theta);
          double m22 = cos(theta);
          double m23 = -cos(phi)*sin(theta);

          double m31 = sin(phi)*cos(theta);
          double m32 = sin(theta);
          double m33 = cos(phi)*cos(theta);

          double xt = m11 * V.x + m12 * V.y + m13 * V.z;
          double yt = (m21 * V.x + m22 * V.y + m23 * V.z);
          double zt = m31 * V.x + m32 * V.y + m33 * V.z;

          V.x = xt;
          V.y = yt;
          V.z = zt;
        }

    }


	
	public void transform(double[][] matrix){
		
	}
}
