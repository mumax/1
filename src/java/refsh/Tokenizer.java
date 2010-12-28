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

public final class Tokenizer extends StreamTokenizer{

	private char chr;
	
	//______________________________________________________________________________________________
	
	public Tokenizer(Reader r) throws IOException{
		super(r);
		ordinaryChars('0', '9');
                ordinaryChars('-', '-');
		ordinaryChars('/', '/');
                wordChars('0', '9');
                wordChars('-', '_');
                wordChars('/', '/');
		wordChars('*', '*');
                
		commentChar('#');
		quoteChar('\'');
	}
	
	//______________________________________________________________________________________________
	
	/*public void readToken() throws IOException{
		chr = (char)nextToken();
	}*/
	
	//______________________________________________________________________________________________
	
	public String currentToken(){
		if(sval == null)
			return (char)ttype + "";
		else
			return sval;
	}
	
	//______________________________________________________________________________________________
	
	public int currentType(){
		return ttype;
	}
	
	//______________________________________________________________________________________________
	
	public static boolean isQuoted(int tokentype){
		return tokentype == '\'' || tokentype == '"';
	}
	
	//______________________________________________________________________________________________
	
	/**
	 * Tokenizer is End-Of-File ?
	 */
	public boolean isEOF(){
		return ttype==TT_EOF;
	}
	//______________________________________________________________________________________________
	
	/**
	 * Tokenizer is End Of Line?
	 */
	public boolean isEOL(){
		return ttype==TT_EOL || isEOF() || ttype==';';
	}
	//______________________________________________________________________________________________
}
