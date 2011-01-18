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
import java.lang.reflect.*;
import java.io.*;

/**
 * Prints text representations of objects and classes for debugging/ general output.
 */
public final class Reflect {

    
    // Debugging is, in general, a cumbersome and tiring task.
    // Wikipedia.
    
    
    /**
     * Prints the values of all fields of the object.
     */
    public static void printObject(Object obj, PrintStream out){
        out.println(toString(obj) + "{");
        for(Class clazz = obj.getClass(); clazz != null; clazz = clazz.getSuperclass()){
            Field[] fields = clazz.getDeclaredFields();
            try{
                AccessibleObject.setAccessible(fields,true);
            }
            catch(SecurityException e){
            }
            for(Field field: fields){
                if(!Modifier.isStatic(field.getModifiers())){
                    out.print("\t" + field.getName());
                    try {
                        Object value = field.get(obj);
                        out.print(" = ");
                        printObjectShort(value, out);
                        out.println();
                    }
                    catch (IllegalArgumentException ex) {
                        throw new Bug();
                    }
                    catch (IllegalAccessException e2) {
                        out.println(" : access denied");
                    }
                }
            }
            out.println();
        }
        out.println("}");
    }
    
    /**
     * Prints the values of all fields of the object to the standard output.
     */
    public static void printObject(Object obj){
        printObject(obj, System.out);
    }
    
    public static void printClass(Class clazz, PrintStream out){
        out.println(clazz.getName() + "{");
        Field[] fields = clazz.getDeclaredFields();
        try{
            AccessibleObject.setAccessible(fields,true);
        } 
        catch(SecurityException e){
        }
        for(Field field: fields){
            if(Modifier.isStatic(field.getModifiers())){
                out.print("\t" + field.getName());
                try {
                    Object value = field.get(null);
                    out.print(" = ");
                    printObjectShort(value, out);
                    out.println();
                } catch (IllegalArgumentException ex) {
                    throw new Bug();
                } catch (IllegalAccessException e2) {
                    out.println(" : access denied");
                }
            }
        }
        
        out.println("}");
    }
    
    public static void printClass(Class clazz){
        printClass(clazz, System.out);
    }
    
    public static void printObjectShort(Object obj, PrintStream out){
        if(obj == null)
            out.print(obj);
        else if(obj.getClass().isArray()){
            out.print(toString(obj) + "{");
            int length = Array.getLength(obj);
            if(length != 0)
                printObjectShort(Array.get(obj, 0), out);
            for(int i=1; i<length; i++){
                out.print(", "); 
                printObjectShort(Array.get(obj, i), out);
            }
            out.print("}");
        }
        else{
            out.print(obj);
        }
    }
    
    /**
     * Creates an easily readible String representation for the Object.
     */
    public static synchronized String toString(Object obj){
	if(obj == null){
	    return "null";
	} else{
	    return classString(obj.getClass()) + createNumber(obj);
	}
    }
    
    private static int createNumber(Object obj){
	return System.identityHashCode(obj);
    }
    
    /**
     * Mooie String voor een klasse, ook array's worden goed weergegeven.
     */
    private static String classString(Class clazz){
	
	if(clazz.isArray())
	    return classString(clazz.getComponentType()) + "[]";
	else
	    return crop(clazz.getName());
    }
    
    
    /**
     * Removes the package prefix
     */
    private static String crop(String className){
	
	int i = className.length()-1;
	while(i > -1 && className.charAt(i) != '.')
	    i--;
	return className.substring(++i, className.length());
    }
}
