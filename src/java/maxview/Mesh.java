import java.awt.*;
import java.awt.geom.*;

/**
Convex 3D object. Niet-convexe objecten kunnen gemaakt worden met Group.
 */
public final class Mesh extends Brush{
	
	private Vertex[] vertex;    //Punten van de 3D-figuur.
    private int[][] polys;      //Lijsten met nummers van vertices die telkens een zijvlak definieren.
    private Color[] fill;       //Kleur zijvlakken
    private Color[] line;       //lijnkleur zijvlakken
    private Polygon[] polys2D;  //Zijvlakken
    private int nPolys;         //Aantal zijvlakken  
    private boolean show[];     //Welke zijvlakken moeten getekend worden op basis van heliciteit.
    private double z;           //verst verwijderde z coordinaat (na transformatie) voor ordening.
    
	
	public Mesh(Vertex[] v, int[][] polys, Color[] fill, Color[] line){
		vertex = v;
		this.polys = polys;
		this.fill = fill;
		this.line = line;
		nPolys = polys.length;
		initPolys2D();
	}
	
	public Mesh(Vertex[] v, int[][] polys, Color fill, Color line){
		this(v, polys, 
			 colorArray(fill, polys.length), colorArray(line, polys.length));
	}
	
	public Mesh(Vertex[] v, int[][] polys){
		this(v, polys, (Color)null, (Color)null);
	}
	
	public Vertex[] getVertices(){
		return vertex;
	}
    
    public void light(Universe universe){
		for(int p = 0; p < nPolys; p++){
			//vectorproduct
			Vertex a = vertex[polys[p][0]];
			Vertex b = vertex[polys[p][1]];
			Vertex c = vertex[polys[p][2]];
			//relatieve vectoren:
			double x1 = a.x-b.x, y1 = a.y-b.y, z1 = a.z-b.z;
			double x2 = c.x-b.x, y2 = c.y-b.y, z2 = c.z-b.z;
			//normaalvector op poly:
			double nx = y1*z2-y2*z1;
			double ny = x2*z1-x1*z2;
			double nz = x1*y2-x2*y1;
			
			final Vertex light = universe.getLight();
			final Vertex ref = vertex[polys[p][0]];
			double lx = light.x - ref.x, ly = light.y - ref.y, lz = light.z - ref.z;
			double inprod = lx*nx + ly*ny + lz*nz;
			inprod /= Math.sqrt((lx*lx + ly*ly + lz*lz) * (nx*nx + ny*ny + nz*nz));
			int red = fill[p].getRed();
			int green = fill[p].getGreen();
			int blue = fill[p].getBlue();
			double d = universe.getContrast();
			fill[p] = new Color(light(red, inprod, d), light(green, inprod, d), light(blue, inprod, d), fill[p].getAlpha()); 
		}
	}
	
	private static int light(int color, double inprod, double lighting){
		double l = (1-inprod)/2; //tussen 0 en 1;
		double stay = (1-lighting) * color;
		double remain = (lighting) * l * color;
		return (int)(stay + remain);
	}
	
	public double getZ(){
		return z;
	}
		
	public void paint(View view){
		updatePolys2D();
		Graphics2D g = view.getGraphics2D();									//?
		for(int i = 0; i < nPolys; i++){
			if(show[i]){
				if(fill[i] != null){
					g.setColor(fill[i]);
					g.fill(polys2D[i]);
					g.draw(polys2D[i]); // voor de edges
				}
				if(line[i] != null){
					g.setColor(line[i]);
					g.draw(polys2D[i]);
				}
			}
		}
	}
	
	public void sort(){
		return;
	}
	
	public Brush copy(){
		
		Vertex[] v = new Vertex[vertex.length];
		for(int i = 0; i < vertex.length; i++)
			v[i] = vertex[i].copy();
		
		int[][] p = new int[nPolys][];
		for(int i = 0; i < nPolys; i++){
			p[i] = new int[polys[i].length];
			for(int j = 0; j < p[i].length; j++)
				p[i][j] = polys[i][j];
		}
		
		Color[] f = new Color[fill.length];
		for(int i = 0; i < fill.length; i++)
			if(fill[i] != null)
				f[i] = new Color(fill[i].getRed(), fill[i].getGreen(), fill[i].getBlue(), fill[i].getAlpha());
		
		Color[] l = new Color[line.length];
		for(int i = 0; i < line.length; i++)
			if(line[i] != null)
				l[i] = new Color(line[i].getRed(), line[i].getGreen(), line[i].getBlue(), line[i].getAlpha());   
		
		Mesh copy = new Mesh(v, p, f, l);
		return copy;
	}
	
	
	/** Zet de kleur van alle zijvlakken op het gegeven kleur. De kleur mag null zijn,
	 dan worden de zijvlakken niet ingekleurd */
	public void setFillColor(Color c){
		fill = colorArray(c, nPolys);
	}
	
	/** Zet de kleur van de individuele zijvlakken op de gegeven kleuren. 
	 Een kleur mag null zijn, dan wordt het overeenkomstig zijvlak niet ingekleurd. */
	public void setFillColor(Color[] c){
		fill = c;
	}
	
	/** Zet de omtrek-kleur van alle zijvlakken op het gegeven kleur. De kleur mag null zijn,
	 dan worden de omtrekken van de zijvlakken niet getekend. */
	public void setLineColor(Color c){
		line = colorArray(c, nPolys);
	}
	
	/** Zet de omtrek-kleur van de individuele zijvlakken op de gegeven kleuren. 
	 Een kleur mag null zijn, dan wordt de overeenkomstige omtrek niet getekend. */
	public void setLineColor(Color[] c){
		line = c;
	}
	
	//berekent nieuwe coordinaten en kleuren van de polygonen.
	//moet aangeroepen worden na elke transform door de viewport.
	private void updatePolys2D(){
		z = Double.NEGATIVE_INFINITY;
		for(int p = 0; p < nPolys; p++){
			Polygon poly2D = polys2D[p];
			int[] x = poly2D.xpoints;
			int[] y = poly2D.ypoints;
			int[] vertexNum = polys[p];
			//dingen achter blikveld worden niet getekend.
			boolean inFront = true;
			for(int i = 0; i < poly2D.npoints; i++){
				Vertex v = vertex[vertexNum[i]];
				x[i] = (int) v.tx;
				y[i] = (int) v.ty;
				//voor of achter blikveld?
				if(v.tz < 0)
					inFront = false;
				//verste z voor ordening
				if(v.tz > z)
					z = v.tz;
			}
			if(inFront){
				//vectorproduct voor heliciteit:
				Vertex a = vertex[polys[p][0]];
				Vertex b = vertex[polys[p][1]];
				Vertex c = vertex[polys[p][2]];
				//heliciteit
				double xt1 = a.tx-b.tx, yt1 = a.ty-b.ty;
				double xt2 = c.tx-b.tx, yt2 = c.ty-b.ty;
				double hel = xt1*yt2 - xt2*yt1;
				show[p] = (hel < 0);
			}
			else
				show[p] = false;
		}
	}
	
	//initialiseerd Polygons met juiste aantal punten.
	private void initPolys2D(){
		polys2D = new Polygon[nPolys];
		for(int i = 0; i < nPolys; i++){
			int npoints = polys[i].length;
			polys2D[i] = new Polygon(new int[npoints], new int[npoints], npoints);
		}
		show = new boolean[nPolys];
	}
	
	private static Color[] colorArray(Color c, int length){
		Color[] array = new Color[length];
		for(int i = 0; i < length; i++)
			array[i] = c;
		return array;
	}
	
	private static int[] intArray(int number, int length){
		int[] array = new int[length];
		for(int i = 0; i < length; i++)
			array[i] = number;
		return array;
	}		
}
