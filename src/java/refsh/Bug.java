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
 * A Bug indicates a bug in the program, something that is not the fault of the
 * user (in that case, an IllegalArgumentException is typically thrown).
 * The main method may catch a Bug, print a bug report with a request to
 * mail it to the maintainer and then terminate the program.
 */
public final class Bug extends Error{
    
    public Bug(Throwable t){
    super(t);
    }
    
    public Bug(){
    super();
    }
    
    public Bug(String message){
    super(message);
    }
}
