/*
 *  This file is part of amumag,
 *  a finite-element micromagnetic simulation program.
 *  Copyright (C) 2006-2008 Arne Vansteenkiste
 * 
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 * 
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details (licence.txt).
 */

// Modified for maxview, 2010, Arne Vansteenkiste

// package x;

import java.awt.Color;
import static java.lang.Math.*;

public class ColorMap3D{

    public static Color map(double x, double y, double z){
        double norm = sqrt(x * x + y * y + z * z);
        double invnorm = 1.0/norm;
         x = invnorm * x;
         y = invnorm * y;
         z = invnorm * z;
        double h = atan2(y,x)/2.0/Math.PI;
        double s = 1.0 - abs(z);
        double b = z > 0.0? 1.0: 1.0+z;
        return Color.getHSBColor((float)h,(float)s,(float)b);
    }
}
