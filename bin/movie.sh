#! /bin/bash
#
# This script converts a series of images into a movie
# typical usage:
# movie.sh *.png
#
# @author Arne Vansteenkiste

tmp=movie.tmp
mkdir $tmp
for i in $@; do
if [ ! -e $tmp/$i.jpg ]; then
  echo -en "\033[s"		# save cursor pos
  echo -n $i;  
  echo -en "\033[K"		# erase rest of line
  convert -quality 100 $i $tmp/$i.jpg;
  echo -en "\033[u"		# restore cursor pos
fi
done;
echo
mencoder "mf://movie.tmp/*.jpg" -mf fps=20 -o movie.avi -ovc lavc
rm -r $tmp