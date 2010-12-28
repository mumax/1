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
 * RefSh, reflective shell.
 */
public class RefSh {

		
	/** Console prompt.
	*/
	private String prompt = "amumag> ";
        private Interpreter interpreter; 
	
	//______________________________________________________________________________________________
	
        
	public static void main (String args[]) throws IOException{
		RefSh refsh = new RefSh(new Interpreter(null, null));
		if(args == null || args.length == 0)
			refsh.interactive();
		else{
			//interpret concatenated arguments.
			StringBuilder buffer = new StringBuilder(args[0]);
			for(int i = 1; i < args.length; i++)
				buffer.append(' ' + args[i]);
			refsh.interpretSafe(buffer.toString());
		}
        }
	

	
	//______________________________________________________________________________________________
        
	public RefSh(Interpreter i){
            this.interpreter = i;
	}
		
	//______________________________________________________________________________________________
	
	public void interactive() throws IOException{
		BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
		System.out.print(prompt);
		String input = in.readLine();
		
		while(input != null){
			interpretSafe(input);
			System.out.print(prompt);
			input = in.readLine();
		}
	}
	
	//______________________________________________________________________________________________
	
	public void interpretSafe(String command){
		try{
                    interpreter.run(command);
		}
		catch(SyntaxException se){
			System.err.println("refsh: " + se.getMessage());
		}
		catch(IOException ioe){
			System.err.println("RefSh unexpected exception:");
			ioe.printStackTrace();
		}
		catch(InvocationTargetException ite){
			ite.getCause().printStackTrace();
		}
		catch(IllegalAccessException iae){
			iae.printStackTrace();
		}
	}
}
