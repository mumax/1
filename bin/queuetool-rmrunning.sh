#! /bin/bash

# Tool for cleaning up crashed simulation output
# Looks for "running" simulations in the current directory and
# removes their *.out directories.

files=$(find */running | tr '\n' ' ' | sed 's/\/running//g')
echo rm -rf $files
rm -rf $files
