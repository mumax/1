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
 * This file contains all the (public) includes for libgpukern.
 *
 * @todo put central documentation here
 * @todo make everything asynchronous and add gpu_sync()
 * 
 * @author Arne Vansteenkiste
 * @author Ben Van de Wiele
 */


#include "gpu_fft0.h"
#include "gpu_kernmul.h"
#include "gpu_linalg.h"
#include "gpu_mem.h"
#include "gpu_normalize.h"
#include "gpu_tensor.h"
#include "gpu_torque.h"
#include "gpu_spintorque.h"
#include "gpu_transpose.h"
#include "gpu_zeropad.h"
#include "gpu_properties.h"
#include "gpu_anal.h"
#include "gpu_init.h"
#include "gpu_reduction.h"
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "gpu_anis.h"
#include "gpu_local_contr3.h"
#include "gpu_exch.h"
#include "gpu_temperature.h"
