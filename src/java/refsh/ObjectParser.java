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
 * ObjectParser turns strings into objects of a specified type, it also has the option to import
 * text or binairy files.
 */

import java.io.*;

public final class ObjectParser {
	
	//______________________________________________________________________________________________
	
	public static Object get(String value, int ttype, Class type) throws ParseException{
		if     (type == Character.TYPE) return parseCharacter(value, ttype);
		else if(type == Boolean.TYPE)   return parseBoolean(value, ttype);
		else if(type == Byte.TYPE)	    return parseByte(value, ttype);
		else if(type == Short.TYPE)	    return parseShort(value, ttype);
		else if(type == Integer.TYPE)   return parseInteger(value, ttype);
		else if(type == Long.TYPE)      return parseLong(value, ttype);
		else if(type == Float.TYPE)     return parseFloat(value, ttype);
		else if(type == Double.TYPE)    return parseDouble(value, ttype);
		else return parseObject(value, ttype, type);
	}
	//______________________________________________________________________________________________
	
	/**
	 * Used only to parse non-primitive Objects.
	 */
	private static Object parseObject(String value, int ttype, Class type) throws ParseException{
		//null
		if(!Tokenizer.isQuoted(ttype) && "null".equals(value))
			return null;
		
		//String constructor
		try{
			return type.getConstructor(String.class).newInstance(new Object[]{value});
		}
		catch(Exception e){
			//read from file
			try{
				return ObjectImporter.read(new File(value), type);
			}
			catch(IOException ioe){
				throw new ParseException(value);
			}
		}
	}
	//______________________________________________________________________________________________
	
	private static Character parseCharacter(String value, int ttype) throws ParseException{
		if(ttype != '\'' || value.length() != 1)
			throw new ParseException(value);
		else
			return new Character(value.charAt(0));
	}
	//______________________________________________________________________________________________
	
	private static Boolean parseBoolean(String value, int ttype) throws ParseException{
		if(!Tokenizer.isQuoted(ttype)){
			if("true".equals(value))
				return new Boolean(true);
			else if("false".equals(value))
				return new Boolean(false);
		}
		throw new ParseException(value);
	}
	//______________________________________________________________________________________________
	
	private static Byte parseByte(String value, int ttype) throws ParseException{
		if(Tokenizer.isQuoted(ttype))
			throw new ParseException(value);
		else
			try{
				return new Byte(value);
			}
			catch(NumberFormatException e){
				throw new ParseException(value);
			}
	}
	//______________________________________________________________________________________________
	
	private static Short parseShort(String value, int ttype) throws ParseException{
		if(Tokenizer.isQuoted(ttype))
			throw new ParseException(value);
		else
			try{
				return new Short(value);
			}
			catch(NumberFormatException e){
				throw new ParseException(value);
			}
	}
	//______________________________________________________________________________________________
	
	private static Integer parseInteger(String value, int ttype) throws ParseException{
		if(Tokenizer.isQuoted(ttype))
			throw new ParseException(value);
		else
			try{
				return new Integer(value);
			}
			catch(NumberFormatException e){
				throw new ParseException(value);
			}
	}
	//______________________________________________________________________________________________
	
	private static Long parseLong(String value, int ttype) throws ParseException{
		if(Tokenizer.isQuoted(ttype))
			throw new ParseException(value);
		else
			try{
				return new Long(value);
			}
			catch(NumberFormatException e){
				throw new ParseException(value);
			}
	}
	//______________________________________________________________________________________________
	
	private static Float parseFloat(String value, int ttype) throws ParseException{
		if(Tokenizer.isQuoted(ttype))
			throw new ParseException(value);
		else
			try{
				return new Float(value);
			}
			catch(NumberFormatException e){
				throw new ParseException(value);
			}
	}
	//______________________________________________________________________________________________
	
	private static Double parseDouble(String value, int ttype) throws ParseException{
		if(Tokenizer.isQuoted(ttype))
			throw new ParseException(value);
		else
			try{
				return new Double(value);
			}
			catch(NumberFormatException e){
				throw new ParseException(value);
			}
	}
	//______________________________________________________________________________________________
}

