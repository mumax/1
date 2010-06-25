#ifndef MAIN_H
#define MAIN_H

#include "debug.h"
#include "gputil.h"
#include "kernel.h"
#include "field.h"
#include "timestep.h"
#include "timer.h"
#include "pipes.h"

param* read_param();

void check_param(param *p);

#endif