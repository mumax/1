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

// Adapted for MuMax by Arne Vansteenkiste, 2010

package refsh;
/**
 * Exports an Object to text.
 */
 import java.io.*;
 
public final class ObjectExporter {

	public static void print(Object obj, PrintStream out) throws IOException{
		if(obj == null)
			out.println("null");
		else if(obj instanceof int[])
			print((int[])obj, out);
		else if(obj instanceof double[])
			print((double[])obj, out);
		else if(obj instanceof int[][])
			print((int[][])obj, out);
		else if(obj instanceof double[][])
			print((double[][])obj, out);
		else 
			out.println(obj.toString());
	}
	
	public static void print(Object obj) throws IOException{
		print(obj, System.out);
	}
	
	public static void print(double[] data, PrintStream out){
		for(int i = 0; i < data.length; i++)
			out.println(data[i]);
		
	}
	
	public static void print(int[] data, PrintStream out){
		for(int i = 0; i < data.length; i++)
			out.println(data[i]);
	}
	
	public static void print(double[][] data, PrintStream out) throws IOException{
        for(int i = 0; i < data.length; i++){
            for(int j = 0; j < data[i].length; j++){
                out.print(data[i][j]);
                out.print(' ');
            }
            out.println();
        }
    }
	
    
    public static void print(int[][] data, PrintStream out) throws IOException{
        for(int i = 0; i < data.length; i++){
            for(int j = 0; j < data[i].length; j++){
                out.print(data[i][j]);
                out.print(" ");
            }
            out.println();
        }
    }
	
	public static void print(int[][] data, File f) throws IOException{
		print(data, new PrintStream(new FileOutputStream(f)));
	}
	
	public static void print(double[][] data, File f) throws IOException{
		print(data, new PrintStream(new FileOutputStream(f)));
	}
}
