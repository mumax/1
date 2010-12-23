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


package x;

import amu.core.Interpolator;
import amu.geom.Vector;
import java.awt.Color;

public class ColorMap {

    private Color[] map = new Color[512];
    private Interpolator ip;
    private double min;
    public double max;
    private Color a, b, c;
    
    public ColorMap(){
        this(1);
    }
    
    public ColorMap(double max){
        this(max, Color.BLUE, Color.WHITE, Color.RED);
    }
    
    public ColorMap(double max, Color a, Color b, Color c){
        this(-max, max, a, b, c);
    }
    
    public ColorMap(double min, double max, Color a, Color b, Color c){
        setColorMap(min, max, a, b, c);
    }
    
    public void setColorMap(double min, double max, Color a, Color b, Color c){
        this.a = a;
        this.b = b;
        this.c = c;
        this.min = min;
        this.max = max;
        ip = new Interpolator(min, 0, max, 511);
        for(int i = 0; i < 256; i++)
            map[i] = mix(a, b, i);
        for(int i = 256; i < 512; i++)
            map[i] = mix(b, c, i-256);
    }
    
    public void setValues(double min, double max){
        setColorMap(min, max, a, b, c);
    }
    
    public void setColors(Color a, Color b, Color c){
        setColorMap(getMin(),getMax(), a, b, c);
    }
    
    private Color mix(Color a, Color b, int i){
        int rd = (255*a.getRed()   + (b.getRed()  -a.getRed())   * i) / 255;
        int gn = (255*a.getGreen() + (b.getGreen()-a.getGreen()) * i) / 255;
        int bl = (255*a.getBlue()  + (b.getBlue() -a.getBlue())  * i) / 255;
        return new Color(rd, gn, bl);
    }
    
    public Color get(double value){
        int color = (int)ip.transf(value);
        if(color > 511)
            color = 511;
        else if(color < 0)
            color = 0;
        return map[color];
    }
    
    public Color get(Vector asDouble){
        return get(asDouble.x);
    }

    public double getMin() {
        return min;
    }

    public double getMax() {
        return max;
    }
}
