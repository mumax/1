  MuMax 0.6.x
  (c) Arne Vansteenkiste & Ben Van de Wiele,
      DyNaMat/EELAB Ghent University.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  This is a beta version for testing purposes.



Requirements
------------

* A 64-bit Linux distribution. MuMax has been tested on Ubuntu 10.04 but should be fairly distribution-independent.
* An nVIDIA CUDA capable GPU, preferably of the new 'Fermi' architecture. See http://www.nvidia.com/object/cuda_gpus.html
* nVIDIA's GPU driver version 260.19 or later. See http://developer.nvidia.com/object/cuda_3_2_downloads.html.
* Python is required only if you want to write python input files.



Installation
------------

* MuMax comes with pre-compiled binaries
* Unpack the mumax archive to the destination of your choice. We will call this /home/me/path/to/mumax
* In the unpacked mumax root directory, execute the setup.bash script:
	./setup.bash
  This needs to be repeated only when the mumax directory was moved.
* MuMax can now be run by executing:
	/home/me/path/to/mumax/bin/mumax
* You can edit your hidden ".bashrc" file in your home directory and add this line:
	export PATH=$PATH:/home/me/path/to/mumax/bin
  After you start a new shell you will be able to start MuMax more simply with the command:
	mumax



Re-compiling
------------

MuMax comes with pre-compiled binaries. To re-compile MuMax you will need, at least:
* nVIDIA's CUDA 3.2 toolkit, See: http://developer.nvidia.com/object/cuda_3_2_downloads.html.
* A compiler for the GO programming language and fully set environment. See: http://golang.org/doc/install.html
* A C/C++ compiler
* A java compiler

To recompile MuMax, execute from its root directory:
	make clean depend all



Getting started
---------------

mumax/examples contains a few example input files. They can be written in two formats:
* The native ".in" format, which contains simple textual commands like "msat 800e3". See example.in
* Python (".py"), which uses the same commands as the ".in" format but with python syntax like "msat(800e3)". See example.py

Run the input files with:
	mumax file.in
or
	mumax file.py

A correspondingly named "file.out" directory will be created where all output is saved.

To get a list of command-line arguments, run:
	mumax -help
arguments include, e.g., -cpu to run on CPU, -gpu=N to select one of multiple GPUs, ...




