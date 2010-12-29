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
 * The RefShell interpreter runs commands.
 */

import java.util.Vector;
import java.lang.reflect.*;
import java.io.*;

public final class Interpreter {

	/** Only allow interaction with this class, no restriction when set to null.
	*/
        private Class restrictClass = null;
	
        /** Call methods on this object, call static if null
         */
        private Object restrictObject = null;
        
	/** Imported packages and classes.
	*/
	static final Vector<String> imports = new Vector<String>(1);

	
	//______________________________________________________________________________________________
	
	public Interpreter(Class restrictClass, Object restrictObject){
            this.restrictClass = restrictClass;
            this.restrictObject = restrictObject;
	}
	
	//______________________________________________________________________________________________
	
	public void run(String command) throws SyntaxException, InvocationTargetException, IllegalAccessException, IOException{

        if("".equals(command)){ return; }

		Tokenizer tokenizer = new Tokenizer(new StringReader(command));
		
		while(!tokenizer.isEOF()){
			
			tokenizer.nextToken();
			if(tokenizer.isEOL())
				continue;
				
			Class clazz;
			Method method;
			Object[] arguments;
			File output = null;
				
			if(tokenizer.currentType() != Tokenizer.TT_WORD)				//check if first token is a word
				throw new SyntaxException("Unexpected token: " + tokenizer);
			
			String token = tokenizer.currentToken();						//determine class and put tokenizer to the method name

                        if (restrictClass != null) {
                            clazz = restrictClass;
                        }
                        else {
                            clazz = parseClass(token);
                            tokenizer.nextToken();										//advance tokenizer to the method name
                            token = tokenizer.currentToken();	//for import
                        }

			if(tokenizer.currentType() == Tokenizer.TT_EOF) 				//no method name
				throw new SyntaxException("Method name expected: " + tokenizer);
			
			if(tokenizer.currentType() != Tokenizer.TT_WORD)				//check if the method token is a word
				throw new SyntaxException("Unexpected token: " + tokenizer);
			
			String methodName = token;										//parse later with overload info.
			tokenizer.nextToken();
			
			Vector<String> args = new Vector<String>();						//make list of arguments
			Vector<Integer> ttypes = new Vector<Integer>();
			while(tokenizer.currentType() == Tokenizer.TT_WORD				
				  || Tokenizer.isQuoted(tokenizer.currentType())){
				args.add(tokenizer.currentToken());
				ttypes.add(tokenizer.currentType());
				tokenizer.nextToken();
			}
			
			if(!tokenizer.isEOL()){											//redirect output
				if(">".equals(tokenizer.currentToken())){
					tokenizer.nextToken();
					output = new File(tokenizer.currentToken());
					tokenizer.nextToken();
					if(!tokenizer.isEOL())
						throw new SyntaxException("End of line expected: " + tokenizer);
				}
				else
					throw new SyntaxException("Unexpected token: " + tokenizer);
			}
			
			
			Object[] methargs = parseMethod(clazz, methodName, args, ttypes);
			method = (Method)methargs[0];
			arguments = (Object[])methargs[1];
			
			if(output != null && method.getReturnType().equals(Void.TYPE))
				throw new SyntaxException("Cannot redirect output of a method with void return type.");
			
			Object returned = method.invoke(restrictObject, arguments);
			if(output == null){
				if(!method.getReturnType().equals(Void.TYPE))
					ObjectExporter.print(returned, System.out);
			}
			else
				ObjectExporter.print(returned, new PrintStream(new FileOutputStream(output)));
		}
	}
	
	//______________________________________________________________________________________________
	
	private Object[] parseMethod(Class clazz, String name, Vector<String> args, Vector<Integer> ttypes) throws SyntaxException{
		
		Method[] methods = clazz.getMethods();
		
		//check if method name exists
		{
			int m = 0;
			while(m<methods.length && 
                                //!
				  !(methods[m].getName().equals(name) && !Modifier.isStatic(methods[m].getModifiers())))
				m++;
			if(m==methods.length){
				throw new SyntaxException("No such method: " + clazz.getName() + "." + name);
			}
		}
		
		Vector<Method> correspond = new Vector<Method>(1);
		for(int i = 0; i < methods.length; i++){
			Method method = methods[i];
			if(method.getName().equals(name) 
			   && !Modifier.isStatic(method.getModifiers()) //!
			   && method.getParameterTypes().length == args.size())
				correspond.add(method);
		}
		
		if(correspond.size() == 0){
			throw new SyntaxException("No such method: " + clazz.getName() + "." + name + 
									  "(" + args.size() + " arguments)");
		}
		
		Object[][] correspondArgs = new Object[correspond.size()][args.size()];
		int validCount = 0;
		Vector<Integer> validIndex = new Vector<Integer>();
		
		for(int m=0; m<correspond.size(); m++){
			correspondArgs[m] = parseArgs(args, ttypes, correspond.get(m).getParameterTypes());
			if(correspondArgs[m] != null){
				validIndex.add(new Integer(m));
			}
		}
		
		if(validIndex.size() == 0)
			throw new SyntaxException("Wrong argument type for method " + name);
		else if(validIndex.size() == 1){
			return new Object[]{correspond.get(validIndex.get(0).intValue()),
								correspondArgs[validIndex.get(0).intValue()]};
		}
		else{
			String msg = "Multiple methods match argument types:";
			for(int i=0; i<validIndex.size(); i++)
				msg = msg + "\n" + correspond.get(validIndex.get(i).intValue());
			throw new SyntaxException(msg);
		}
		
	}
	
	//______________________________________________________________________________________________
	
	private Object[] parseArgs(Vector<String> args, Vector<Integer> ttypes, Class[] types){
		Object[] parsed = new Object[args.size()];
		try{
			for(int i=0; i<parsed.length; i++)
				parsed[i] = ObjectParser.get(args.get(i), ttypes.get(i), types[i]);
			return parsed;
		}
		catch(ParseException e){
			return null;
		}
	}

	//______________________________________________________________________________________________
	
	public Class parseClass(String name) throws SyntaxException{
		if(name.contains("."))
			try{
				return Class.forName(name);
			}
			catch(ClassNotFoundException e){
				throw new SyntaxException(e.getMessage());
			}
		else{
			Vector<Class> correspond = new Vector<Class>(1);
			for(int i = 0; i < imports.size(); i++){
				try{
					correspond.add(Class.forName(imports.get(i) + name));
				}
				catch(ClassNotFoundException e){
				}
			}
			if(correspond.size() == 0)
				throw new SyntaxException("Class not found: " + name);
			else if(correspond.size() == 1)
				return correspond.get(0);
			else{
				StringBuffer msg = new StringBuffer(name + "is ambigous, both ");
				for(int i = 0; i < correspond.size()-1; i++)
					msg.append(correspond.get(i).getName() + ", ");
				msg.append(correspond.get(correspond.size()-1) + " match.");
				throw new SyntaxException(msg.toString());
			}
		}
	}
}
