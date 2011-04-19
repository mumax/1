/*
 *  This file is part of MuMax, a high-performance micromagnetic simulator.
 *  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele.
 *  Use of this source code is governed by the GNU General Public License version 3
 *  (as published by the Free Software Foundation) that can be found in the license.txt file.
 *
 *  Note that you are welcome to modify this code under condition that you do not remove any 
 *  copyright notices and prominently state that you modified it, giving a relevant date.
 */

/**
 * @file
 *
 * This file contains all the (public) includes for libcpukern.
 *
 * @todo put central documentation here
 *
 * @author Arne Vansteenkiste
 * @author Ben Van de Wiele
 */


#include "cpu_fft.h"
#include "cpu_kernmul.h"
#include "cpu_linalg.h"
#include "cpu_mem.h"
#include "cpu_normalize.h"
#include "cpu_torque.h"
#include "cpu_transpose.h"
#include "cpu_zeropad.h"
#include "cpu_properties.h"
#include "cpu_anal.h"
#include "cpu_init.h"
#include "cpu_reduction.h"
#include "cpu_local_contr3.h"
#include "thread_functions.h"
#include "cpu_exch.h"
