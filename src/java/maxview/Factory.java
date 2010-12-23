/**
  This file is part of a 3D engine,
  copyright Arne Vansteenkiste 2006-2010.
  Use of this source code is governed by the GNU General Public License version 3,
  as published by the Free Software Foundation.
*/
import java.awt.*;
import static java.lang.Math.sin;
import static java.lang.Math.cos;

public final class Factory {

    /**
     Maakt een kubus met ribbe r.  Het middelpunt van het grondvlak staat in de
	 oorsprong (0, 0, 0) (de figuur kan verplaatst worden met translate(dx, dy, dz)).
	 */
    public static Mesh cube(double r){
        r/=2;
        Vertex[] v = new Vertex[]{
            new Vertex(-r, -r, -r),
            new Vertex(r, -r, -r),
            new Vertex(r, r, -r),
            new Vertex(-r, r, -r),
            new Vertex(-r, -r, r),
            new Vertex(r, -r, r),
            new Vertex(r, r, r),
            new Vertex(-r, r, r)};
        int[][] polys = new int[][]{
		{0, 3, 2, 1},
		{4, 5, 6, 7},
		{0, 1, 5, 4},
		{1, 2, 6, 5},
		{2, 3, 7, 6},
		{0, 4, 7, 3}};
        Mesh cube = new Mesh(v, polys);
        cube.translate(0, r, 0);
        return cube;
    }


    /**
    Maakt een balk met resp breedte, hoogte en diepte x, y en z.
	 Het middelpunt van het grondvlak staat in de
	 oorsprong (0, 0, 0) (de figuur kan verplaatst worden met translate(dx, dy, dz)).
	 */
    public static Mesh beam(double x, double y, double z){
        x/=2;
        y/=2;
        z/=2;
        Vertex[] v = new Vertex[]{
            new Vertex(-x, -y, -z),
            new Vertex(x, -y, -z),
            new Vertex(x, y, -z),
            new Vertex(-x, y, -z),
            new Vertex(-x, -y, z),
            new Vertex(x, -y, z),
            new Vertex(x, y, z),
            new Vertex(-x, y, z)};
        int[][] polys = new int[][]{
		{0, 3, 2, 1},
		{4, 5, 6, 7},
		{0, 1, 5, 4},
		{1, 2, 6, 5},
		{2, 3, 7, 6},
		{0, 4, 7, 3}};
        Mesh beam =  new Mesh(v, polys);
        beam.translate(0, y, 0);
        return beam;
    }


    /**
    Maakt een cylinder met gegeven straal en hoogte.  Het grondvlak is geen echte
	 cirkel maar een regelmatige n-hoek met n gegeven door npoints.
	 Het middelpunt van het grondvlak staat in de
	 oorsprong (0, 0, 0) (de figuur kan verplaatst worden met translate(dx, dy, dz)).
	 */
    public static Mesh cylinder(double r, int npoints, double height){
        height/=2;
        Vertex[] v = new Vertex[npoints * 2];
        for(int i = 0; i < npoints*2; i++){
            double theta = (-2*Math.PI/npoints)*i;
            v[i] = new Vertex(r*Math.cos(theta), (i<npoints?-1:1)*height, r*Math.sin(theta));
        }
        int npoly = npoints + 2;
        int[][] p = new int[npoly][];
        for(int i = 0; i < npoints; i++){
            p[i] = new int[4];
            for(int j=0; j<4; j++){
                p[i][0] = i;
                p[i][1] = (i+1) % npoints;
                p[i][2] = ((i+1) % npoints) + npoints;
                p[i][3] = i+npoints;
            }
        }
        int bottom = npoly - 2;
        int top = npoly-1;
        p[bottom] = new int[npoints];
        p[top] = new int[npoints];

        for(int i = 0; i < npoints; i++){
            p[top][i] = i + npoints;
            p[bottom][i] = npoints - 1 - i;
        }
        Mesh cyl = new Mesh(v, p);
        cyl.translate(0, height, 0);
        return cyl;
    }


    /**
    Maakt een kegel met gegeven straal en hoogte.  Het grondvlak is geen echte
	 cirkel maar een regelmatige n-hoek met n gegeven door npoints.
	 Het middelpunt van het grondvlak staat in de
	 oorsprong (0, 0, 0) (de figuur kan verplaatst worden met translate(dx, dy, dz)).
	 pivot vertex in centrum
	 */
    public static Mesh cone(double r, int npoints, double height){
        Vertex[] v = new Vertex[npoints + 1 + 1];

        for(int i = 0; i < npoints; i++){
            double theta = (-2*Math.PI/npoints)*i;
            v[i] = new Vertex(r*Math.cos(theta), 0, r*Math.sin(theta));
        }
        v[npoints] = new Vertex(0, height, 0); //top
        v[npoints+1] = new Vertex(0, height/2, 0); //pivot
        int npoly = npoints + 1;
        int[][] p = new int[npoly][];
        for(int i = 0; i < npoints; i++){
            p[i] = new int[3];
            for(int j=0; j<4; j++){
                p[i][0] = i;
                p[i][1] = (i+1) % npoints;
                p[i][2] = npoints;
            }
        }
        int bottom = npoly - 1;
        p[bottom] = new int[npoints];

        for(int i = 0; i < npoints; i++){
            p[bottom][i] = npoints - 1 - i;
        }
        Mesh cone = new Mesh(v, p);
        cone.translate(0, -height/2, 0);
        return cone;
    }

  /**
   Rotated cone
  */
  public static Mesh cone(double r, int npoints, double height, double theta, double phi){
        Vertex[] v = new Vertex[npoints + 1 + 1];

        for(int i = 0; i < npoints; i++){
            double theta_ = (-2*Math.PI/npoints)*i;
            v[i] = new Vertex(r*Math.cos(theta_), 0, r*Math.sin(theta_));
        }
        v[npoints] = new Vertex(0, height, 0); //top
        v[npoints+1] = new Vertex(0, height/2, 0); //pivot
        int npoly = npoints + 1;
        int[][] p = new int[npoly][];
        for(int i = 0; i < npoints; i++){
            p[i] = new int[3];
            for(int j=0; j<4; j++){
                p[i][0] = i;
                p[i][1] = (i+1) % npoints;
                p[i][2] = npoints;
            }
        }
        int bottom = npoly - 1;
        p[bottom] = new int[npoints];

        for(int i = 0; i < npoints; i++){
            p[bottom][i] = npoints - 1 - i;
        }

        for(int i=0; i<v.length; i++){
          Vertex V = v[i];
          V.y -= height / 2;

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
        Mesh cone = new Mesh(v, p);
        return cone;
    }



    /**
    maakt een ellipsoide met straal r in het xz vlak en straal height in de y
	 richting.
	 */
    public static Mesh sphere(double r, int hpoints, int vpoints){
        vpoints--;  //eerst boven/onder open laten en later opvullen met kegels
        vpoints = 2*vpoints;
        int npoints = hpoints*vpoints;
        Vertex[] vert = new Vertex[npoints];
        for(int v = 0; v < vpoints; v++)
            for(int h = 0; h < hpoints; h++){
                double theta = (-2*Math.PI/hpoints)*h;
                double y = (2*r/(vpoints+1))*v - r;
                double rho = Math.sqrt(r*r-y*y);
                double x = rho*Math.cos(theta);
                double z = rho*Math.sin(theta);
                vert[index(h, v, hpoints)] = new Vertex(x, y, z);
            }

				int[][] poly = new int[hpoints*(vpoints-1) + 2][];
        int p=0;
        for(int v = 0; v < vpoints-1; v++)
            for(int h = 0; h < hpoints; h++, p++){
                poly[p] = new int[4];
                poly[p][0] = index(h, v, hpoints);
                poly[p][1] = index((h+1)%hpoints, v, hpoints);
                poly[p][2] = index((h+1)%hpoints, v+1, hpoints);
                poly[p][3] = index(h, v+1, hpoints);
			}

				int bottom = hpoints*(vpoints-1) + 1;
        int top = hpoints*(vpoints-1);
        poly[bottom] = new int[hpoints];
        poly[top] = new int[hpoints];

        for(int i = 0; i < hpoints; i++){
            poly[top][i] = i + npoints-hpoints;
            poly[bottom][i] = hpoints - 1 - i;
        }

        return new Mesh(vert, poly);
    }

    //voor bol
    private static int index(int h, int v, int hpoints){
        return h + v * hpoints;
    }

    //voor afplatten polen van de bol.
    public static double PLAT = 0.00;
    private static double offset(int v, int vpoints, double r){
        double offset = (r / vpoints) * PLAT;
        if(v == 0){
            return offset;
        }
        else if(v == vpoints-1){
            return -offset;
        }
        else
            return 0;
    }

    /**
		maakt een rechthoek in het xz vlak met breedte en diepte x en y.
	 */
    public static Mesh sheet(double x, double y){

		x/= 2;
		y/= 2;

		Vertex[] v = new Vertex[4];
		v[0] = new Vertex(-x, 0, -y);
		v[1] = new Vertex(x, 0, -y);
		v[2] = new Vertex(x, 0, y);
		v[3] = new Vertex(-x, 0, y);

		int[][] p = new int[][]{{0, 3, 2, 1}};

		return new Mesh(v, p);
    }
}
