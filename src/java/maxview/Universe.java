import java.awt.Color;
import java.awt.Polygon; 
import java.awt.Graphics2D;

/**
Een Universe bevat een root Group van 3D Objecten en belichting.
*/
public final class Universe{
	
    //Achtergrondkleur van het Universe.
    private Color background = Color.BLACK;
    
	//positie van de lichtbron.
    public Vertex light;  
    
	//sterkte van de belichting.
	public double lighting = 0;                
	
	//root group
	private Group root;
	
    public Universe(Color background, Vertex light, double contrast){
        this.background = background;
		this.light = light;
		this.lighting = contrast;
    }
	
	public double getContrast(){
		return lighting;
	}
	
	public void setRoot(Group root){
		this.root = root;
		root.light(this);
		root.sort();
	}
	
	public Group getRoot(){
		return root;
	}
	
	public Color getBackground(){
		return background;
	}

	public Vertex getLight(){
		return light;
	}
    
    /**
	 Zet de positie van het licht. (heeft maar zin als lighting > 0).
	 Dit moet gebeuren vooraleer een Brush toegevoegd wordt, anders wordt
	 deze belicht met de oude licht-positie.
	 */
    public void setLight(double x, double y, double z){
        light.x = x;
        light.y = y;
        light.z = z;
    }
}
