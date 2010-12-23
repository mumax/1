import java.io.Serializable;
import java.awt.Color;

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
	
	public void transform(double[][] matrix){
		
	}
}
