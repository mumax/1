import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

/**
View voor een universe. Bevat de camera-positie en -richting.
*/
public final class View extends JPanel{
	
    //Het universum dat deze viewport toont.
    private Universe universe;
    
    //Perspectief parameter: bepaalt hoe klein de dingen er uit zien.
    private double persp = 1000;
    
    //Camera positie.
    private double camx, camy , camz;
    private double phi, theta;					//0..2pi, -pi/2..pi/2
    private double bordr, bordphi, bordtheta;   //0..inf, 0..2pi, 0..pi/2
	
	//Huidige schermafmetingen
	private int width, height;
	
    //Kleur van de cursor;
    private Color crosshair = new Color(0, 0, 255, 200);
    
    //transformatiematrix
    private double m11, m12, m13;
    private double m21, m22, m23;
    private double m31, m32, m33;
	
	//Graphics2D
	private Graphics2D g;
	
	//Anti-alias
	private boolean antialias = true;
	
	
    public View(Universe u){
		g = (Graphics2D)getGraphics();
        universe = u;
		setFocusable(true);
		addKeyListener(new PortKeyListener());
		//addMouseListener(new PortMouseListener());
		//addMouseMotionListener(new PortMouseMotionListener());
    }
	
	public void setAntiAlias(boolean a){
		this.antialias = a;
	}
	
	public void setBordRadius(double r){
		setBord(r, bordphi, bordtheta);
	}
	
	public void setBordPhi(double bphi){
		setBordDirection(bphi, bordtheta);
	}
	
	public void setBordTheta(double btheta){
		setBordDirection(bordphi, btheta);
	}
	
	public void setBordDirection(double bphi, double btheta){
		setBord(bordr, bphi, btheta);
	}
	
	public void setBord(double r, double bphi, double btheta){
		bordr = r;
		bordphi = bphi;
		bordtheta = btheta;
		
		camx = r*cos(btheta)*sin(bphi);
		camy = r*sin(btheta);
		camz = r*cos(btheta)*cos(bphi);
		phi = (bphi+ PI) % (2*PI);
		theta = -btheta;
		
		System.out.println("setBord: r=" + r + ", bphi=" + bphi + ", btheta=" + btheta);
		System.out.println("camx=" + camx + ", camy=" + camy + ", camz=" + camz);
		System.out.println("theta=" + theta + ", phi=" + phi);
		initMatrix();
	}
	
	public void moveCamera(double dx, double dy, double dz){
        setCameraPosition(camx + dx, camy + dy, camz + dz);
    }
    
    public void setCameraPosition(double x, double y, double z){
        camx = x;
        camy = y;
        camz = z;
		System.out.println("camx=" + camx + ", camy=" + camy + ", camz=" + camz);
		initMatrix();
    }
    
    public void rotateCamera(double dPhi, double dTheta){
        phi += dPhi;
        phi %= 2*PI;
        theta += dTheta;
        if(theta > PI/2)
            theta = PI/2;
        else if(theta < -PI/2)
            theta = -PI/2;
        setCameraDirection(phi, theta);
    }
    
    public void setCameraDirection(double phi, double theta){
        this.phi = phi;
        this.theta = theta;
		System.out.println("theta=" + theta + ", phi=" + phi);
		initMatrix();
    }
	
    //Transformeert alle vertices naar huidige camera stand.
    private void transform(Vertex[] vertex){
        for(int i = 0; i < vertex.length; i++)
            transform(vertex[i]);
    }
    
    public Graphics2D getGraphics2D(){
		return g;
	}
	
    private void initMatrix(){
        m11 = cos(phi);
        m12 = 0;
        m13 = -sin(phi);
        
        m21 = -sin(phi)*sin(theta);
        m22 = cos(theta);
        m23 = -cos(phi)*sin(theta);
        
        m31 = sin(phi)*cos(theta);
        m32 = sin(theta);
        m33 = cos(phi)*cos(theta);
    }
    
    
    private void transform(Vertex v){
        //Translatie naar camera
        double x = v.x - camx;
        double y = v.y - camy;
        double z = v.z - camz;
        
        //Rotatie
        double xt = m11 * x + m12 * y + m13 * z;
        double yt = -(m21 * x + m22 * y + m23 * z);
        double zt = m31 * x + m32 * y + m33 * z;
        
        //Aanpassen aan scherm + perspectief
        v.tx = ((xt/zt)*persp) + width/2;
        v.ty = ((yt/zt)*persp) + height/2;
        v.tz = zt;
    }
	
	public void paint(Graphics g1){
		g = (Graphics2D)g1;
		if(antialias)
			g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
		else
			g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);
		Dimension d = getSize();
		width = d.width;
		height = d.height;
		g.setColor(universe.getBackground());
		g.fillRect(0, 0, width, height);
		
		transform(universe.getRoot().getVertices());
		universe.getRoot().paint(this);
		
		g.setColor(crosshair);
		//light tekenen
		transform(universe.light);
		int lx = (int)universe.light.tx;
		int ly = (int)universe.light.ty;
		g.drawRect(lx, ly, 3, 3);
		
		//oorsprong
		Vertex origin = new Vertex(0, 0, 0);
		transform(origin);
		int ox = (int)origin.tx;
		int oy = (int)origin.ty;
		int C = 8;
		g.drawLine(ox, oy+C,ox, oy-C);
		g.drawLine(ox+C, oy, ox-C, oy);
	} 
	
	/*public void update(Graphics g1){
		paint(g1);
	}*/
	
	public static final double PI = Math.PI;
	public static final double DEG = Math.PI / 180;
	
	private double cos(double a){
		return Math.cos(a);
	}
	
	private double sin(double a){
		return Math.sin(a);
	}
	
	final class PortKeyListener implements KeyListener{
		
		//Hoeveel bewogen wordt per key-press.
		private int DELTA = 10;
		
		private static final double DPHI = 0.1, DTHETA = 0.1, DR = 1, D=10;
		///*
		public void keyPressed(KeyEvent e){
			int key = e.getKeyCode();
			switch(key){
				case KeyEvent.VK_LEFT: setBordPhi(bordphi+DPHI); repaint(); break;
				case KeyEvent.VK_RIGHT: setBordPhi(bordphi-DPHI); repaint(); break; //rotateWorld(-PI/128); repaint(); break;
				case KeyEvent.VK_UP: setBordTheta(bordtheta+DTHETA); repaint(); break;
				case KeyEvent.VK_DOWN: setBordTheta(bordtheta-DTHETA); repaint(); break;
				case KeyEvent.VK_C: setBordRadius(bordr-DR); repaint(); break;
				case KeyEvent.VK_SPACE: setBordRadius(bordr+DR); repaint(); break;
				default: break;
			}
		}
		//*/
		/*
		public void keyPressed(KeyEvent e){
			int key = e.getKeyCode();
			switch(key){
				case KeyEvent.VK_LEFT: moveCamera(-D, 0, 0); repaint(); break;
				case KeyEvent.VK_RIGHT: moveCamera(D, 0, 0); repaint(); break; //rotateWorld(-PI/128); repaint(); break;
				case KeyEvent.VK_UP: moveCamera(0, 0, D); repaint(); break;
				case KeyEvent.VK_DOWN: moveCamera(0, 0, -D); repaint(); break;
				case KeyEvent.VK_C: moveCamera(0, -D, 0); repaint(); break;
				case KeyEvent.VK_SPACE: moveCamera(0, D, 0); repaint(); break;
				default: break;
			}
		}//*/
		public void keyReleased(KeyEvent e){}
		public void keyTyped(KeyEvent e){}
	}
	
	//waar muis is tijdens draggen
	private int downX, downY;
	
	//Hoe gevoelig kijken met de muis is.
	public double mouseSensitivity = 0.002;
	
	final class PortMouseListener implements MouseListener{
		public void mousePressed(MouseEvent e){
			downX = e.getX();
			downY = e.getY();
		}
		public void mouseReleased(MouseEvent e){}
		public void mouseClicked(MouseEvent e){}
		public void mouseEntered(MouseEvent e){}
		public void mouseExited(MouseEvent e){}
	}
	
	final class PortMouseMotionListener implements MouseMotionListener{
		public void mouseMoved(MouseEvent e){}
		
		public void mouseDragged(MouseEvent e){
			int upX = e.getX();
			int upY = e.getY();
			rotateCamera((upX-downX)*mouseSensitivity, 
						 -(upY-downY)*mouseSensitivity);
			downX = upX;
			downY = upY;
			repaint();
		}
	}
}