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
import java.io.*;
import java.util.*;

/**
 * Reader voor textbestanden met een typische array-indeling.
 */
public final class ObjectImporter extends BufferedReader{

	/**
		Commentaar teken, alle lijnen die hier mee beginnen worden genegeerd.
	 */
	public static final char COMMENT = '#';
	
    private int chr; // buffer: leest steeds 1 char vooruit.
	
	public static Object read(File file, Class type) throws IOException, ParseException{
		ObjectImporter in = new ObjectImporter(file);
		if(type.equals(double[][].class)){
			double[][] obj = in.readDoubleArray();
			in.close();
			return obj;
		}
		else if(type.equals(double[].class)){
			double[] obj = in.readDoubleList();
			in.close();
			return obj;
		}
		else
			throw new ParseException();
	}
	
	public ObjectImporter(File file) throws IOException{
		this(new FileInputStream(file));
	}
	
    public ObjectImporter(Reader in) throws IOException{
        super(in);
		mark(0);
        chr = read();
    }
	
	public ObjectImporter(InputStream in) throws IOException{
		this(new InputStreamReader(in));
	}
    
	/**
		is End-Of-File ?
	 */
	public boolean isEOF(){
		return chr == -1;
	}
	
	/**
		is End Of Line?
	 */
	public boolean isEOL(){
		return chr == '\n' || isEOF();
	}
	
	
	/**
	 Leest woord:
	 Comment char? sla lijn over.
	 Sla spaties over.
	 Einde lijn? return null.
	 anders: geef woord, sla opniew spaties over -> staat evt. klaar op EOL().
	*/
    public String readString() throws IOException{
		while(chr == COMMENT)
			skipLine();
		skipSpaces();
		if(isEOL())
			return null;
		else{
			StringBuffer buffer = new StringBuffer();
			while(!isWhitespace(chr)){
				buffer.append((char)chr);
				chr = read();
			}
			skipSpaces();
			return buffer.toString();
		}
    }
	
	public int readInt() throws IOException{
		return Integer.parseInt(readString());
	}
	
	public double readDouble() throws IOException{
		return Double.parseDouble(readString());
	}
	
	
	/*public int[] readIntList() throws IOException{
		int[] list = new int[wordCount()];
		for(int i = 0; i < list.length; i++)
			list[i] = readInt();
		return list;
	}
	
	public double[] readDoubleList() throws IOException{
		double[] list = new double[wordCount()];
		for(int i = 0; i < list.length; i++)
			list[i] = readDouble();
		return list;
	}
	
	public int[] readIntColumn(int column) throws IOException{
		int[] list = new int[lineCount()];
		for(int i = 0; i < list.length; i++){
			for(int w = 0; w < column; w++)
				skipWord();
			list[i] = readInt();
			skipLine();
		}
		return list;
	}
	
	public double[] readDoubleColumn(int column) throws IOException{
		double[] list = new double[lineCount()];
		for(int i = 0; i < list.length; i++){
			for(int w = 0; w < column; w++)
				skipWord();
			list[i] = readDouble();
			skipLine();
		}
		return list;
	}*/
	
	
	public double[] readDoubleColumn(int col) throws IOException{
		DoubleArray list = new DoubleArray();
		while(!isEOF()){
			for(int i = 0; i < col; i++)
				skipWord();
			list.add(readDouble());
			skipLine();
		}
		return list.toArray();
	}
	
	public double[] readDoubleList() throws IOException{
		return readDoubleColumn(0);
	}
	
	public int[] readIntColumn(int col) throws IOException{
		IntArray list = new IntArray();
		while(!isEOF()){
			for(int i = 0; i < col; i++)
				skipWord();
			list.add(readInt());
			skipLine();
		}
		return list.toArray();
	}
	
	public int[][] readIntArray() throws IOException{
		ArrayList<int[]> lines = new ArrayList<int[]>();
		IntArray words = new IntArray();
		while(!isEOF()){
			while(!isEOL()){
				words.add(readInt());
			}
			lines.add(words.toArray());
			words.clear();
			skipLine();
		}
		int[][] array = new int[lines.size()][];
		for(int i = 0; i < array.length; i++)
			array[i] = lines.get(i);
		return array;
	}
	
	
	public double[][] readDoubleArray() throws IOException{
		ArrayList<double[]> lines = new ArrayList<double[]>();
		DoubleArray words = new DoubleArray();
		while(!isEOF()){
			while(!isEOL()){
				words.add(readDouble());
			}
			lines.add(words.toArray());
			words.clear();
			skipLine();
		}
		double[][] array = new double[lines.size()][];
		for(int i = 0; i < array.length; i++)
			array[i] = lines.get(i);
		return array;
	}
	
	
	public void skipWord() throws IOException{
		while(chr == COMMENT)
			skipLine();
		skipSpaces();
		while(!isWhitespace(chr))
			chr = read();
		skipSpaces();
	}
	
	public void skipLine() throws IOException{
		while(chr != '\n')
			chr = read();
		chr = read();
	}
	
		
	private void skipSpaces() throws IOException{
		while(!isEOF() && isSpace(chr)){
			chr = read();
		}
	}
	
	/** Is spatie maar geen newline? */
	private boolean isSpace(int c){
		return Character.isWhitespace((char)c) && ((char)c) != '\n';
	}
	
	private boolean isWhitespace(int c){
		return Character.isWhitespace((char)c);
	}
	
	public static String readFile(File file) throws IOException{
		FileInputStream in = new FileInputStream(file);
		byte[] bytes = new byte[(int)(file.length())];
		in.read(bytes);
		return new String(bytes);
	}
	
	public final class DoubleArray {
    
    private double[] data;
    private int size;
    
    public DoubleArray(){
        this(10);
    }
    
    public DoubleArray(int capacity){
        if(capacity < 1)
            throw new IllegalArgumentException("Vector capacity < 1");
        data = new double[capacity];
        size = 0;
    }
    
    public int size(){
        return size;
    }
    
    public void add(double d){
        size++;
        if(size == data.length){
            double[] buffer = new double[2*data.length];
            for(int i = 0; i < data.length; i++)
                buffer[i] = data[i];
            data = buffer;
        }
        data[size-1] = d;
    }
    
    public double get(int index){
        return data[index];
    }
	
	public void set(int index, double d){
		if(index >= size)
			throw new ArrayIndexOutOfBoundsException();
		else
			data[index] = d;
	}
	
	public void clear(){
		size = 0;
	}	
	
	public double[] toArray(){
		double[] buffer = new double[size()];
		for(int i = 0; i < buffer.length; i++){
			buffer[i] = data[i];
		}
		return buffer;
	}
}



public final class IntArray {
    
    private int[] data;
    private int size;
    
    public IntArray(){
        this(10);
    }
    
    public IntArray(int capacity){
        if(capacity < 1)
            throw new IllegalArgumentException("Vector capacity < 1");
        data = new int[capacity];
        size = 0;
    }
    
    public int size(){
        return size;
    }
    
    public void add(int d){
        size++;
        if(size == data.length){
            int[] buffer = new int[2*data.length];
            for(int i = 0; i < data.length; i++)
                buffer[i] = data[i];
            data = buffer;
        }
        data[size-1] = d;
	}
	
	public int get(int index){
		return data[index];
	}
	
	public void set(int index, int d){
		if(index >= size)
			throw new ArrayIndexOutOfBoundsException();
		else
			data[index] = d;
	}
	
	public void clear(){
		size = 0;
	}
	
	public int[] toArray(){
		int[] buffer = new int[size()];
		for(int i = 0; i < buffer.length; i++){
			buffer[i] = data[i];
		}
		return buffer;
	}
}




}