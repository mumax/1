#! /bin/bash
#
# This script converts a series of images into a movie
# typical usage:
# movie.sh *.png
#
# @author Arne Vansteenkiste

for i in $@; do
if [ ! -e $i.jpg ]; then
  echo $i;
  convert -quality 100 $i $i.jpg;
fi
done;
mencoder "mf://*.jpg" -mf fps=20 -o movie.avi -ovc lavc
