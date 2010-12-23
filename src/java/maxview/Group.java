import java.awt.Color;

/**
Groep van Brushes die afzonderlijk als convex beschouwd worden. Als de Brushes
niet-convex zijn, mogen ze niet overlappen. 
De totale groep wordt weer als convex beschouwd en mag niet overlappen met een
andere groep, anders moeten de twee samengevoegd worden.
*/
public final class Group extends Brush{

	//Vertex grid
	private Vertex[] vertices;
	private int nVertex;
	
	//children
	protected Brush[] brushes; 
	protected int nBrush;
	
	private boolean sorted = true;
	
	public Group(){
		this(true);
	}
	
	public Group(boolean sorted){
		brushes = new Brush[1];
		vertices =  new Vertex[1];
		nVertex = 0;
		nBrush = 0;
		this.sorted = sorted;
	}
	
	public Vertex[] getVertices(){
		return vertices;
	}
	
	public void light(Universe u){
		for(int i = 0; i < nBrush; i++)
			brushes[i].light(u);
	}
	
	public double getZ(){
		double z = brushes[0].getZ();
		for(int i = 1; i < nBrush; i++){
			final double zbuffer = brushes[i].getZ();
			if(zbuffer > z)
				z = zbuffer;
		}
		return z;
	}
	
	public void paint(View v){
		// nog occlusie
		sort();
        for(int i = 0; i < nBrush; i++)
            brushes[i].paint(v);
    }
    
	public void sort(){
		if(sorted){
			for(int i = 0; i < nBrush; i++)
				brushes[i].sort();
			//i wordt evt naar voor geschoven.
			for(int i = 1; i < nBrush; i++){
				int pos = i;
				while(pos > 0 && brushes[pos].getZ() > brushes[pos-1].getZ()){
					Brush buffer = brushes[pos];
					brushes[pos] = brushes[pos-1];
					brushes[pos-1] = buffer;
					pos--;
				}
			}
		}
		else
			return;
    }
	
	public Brush copy(){
		Group g = new Group();
		for(int i = 0; i < nBrush; i++)
			g.add(brushes[i].copy());
		return g;
	}
	
	public void setFillColor(Color c){
		for(int i = 0; i < nBrush; i++)
			brushes[i].setFillColor(c);
	}
	
	public void setLineColor(Color c){
		for(int i = 0; i < nBrush; i++)
			brushes[i].setLineColor(c);
	}
		
    public void add(Brush b){
        if(nBrush < brushes.length){                //Kan er nog bij.
            brushes[nBrush] = b;
            nBrush++;
        }
        else{                                       //Moet uitbreiden.
            Brush[] newBrush = new Brush[nBrush*2];
            for(int i=0; i<nBrush; i++)
                newBrush[i] = brushes[i];
            brushes = newBrush;
            brushes[nBrush] = b;
            nBrush++;
        }
		addVertices(b.getVertices());
    }
	
	private void addVertices(Vertex[] v){
		for(int i = 0; i < v.length; i++)
			addVertex(v[i]);
	}
	
	private void addVertex(Vertex v){
		if(nVertex < vertices.length){                //Kan er nog bij.
            vertices[nVertex] = v;
            nVertex++;
        }
        else{                                       //Moet uitbreiden.
            Vertex[] newVertex = new Vertex[nVertex+1]; // aanpassen !!
            for(int i=0; i < nVertex; i++)
                newVertex[i] = vertices[i];
            vertices = newVertex;
            vertices[nVertex] = v;
            nVertex++;
        }		
	}
}
