import java.io.Serializable;

/*  
BarneX 3D Engine.  Copyright Arne Vansteenkiste 2006.
*/
public final class Vertex implements Serializable{
	
    //Coordinaten in het universum.
    public double x, y, z;
    
    //Getransformeerde Coordinaten.
    public double tx, ty, tz;
    

    public Vertex(double x, double y, double z){
        this.x = x;
        this.y = y; 
        this.z = z;
    }
    
    
    public void translate(double dx, double dy, double dz){
        x += dx;
        y += dy;
        z += dz;
    }
    
    
    public Vertex copy(){
        return new Vertex(x, y, z);
    }
}
