#! /bin/bash
#
# This utility periodically checks the subversion status of the current directory
# and displays a pop-up notification when a new update is available.
# 
# Usage:
#
# First make sure libnotify is installed:
# sudo apt-get install libnotify-bin
# then run:
# watchsvn.sh &
#
INTERVAL=120 #check every X seconds.


(for (( ;; )); do

  LOCAL=$(svn info | grep "Revision: " | tr -d "Revision: ");
  REMOTE=$(svn stat -uq | grep "Status against revision:" | tr -d "Status against revision:\t");

  if ((( $LOCAL != $REMOTE ))); then
#    echo "New SVN update available: revision" $REMOTE
    notify-send "New SVN update available: " $REMOTE
    while ((( $LOCAL != $REMOTE ))); do
      sleep $INTERVAL
      
      LOCAL=$(svn info | grep "Revision: " | tr -d "Revision: ");
      REMOTE=$(svn stat -uq | grep "Status against revision:" | tr -d "Status against revision:\t");

    done;
  else
#    echo up to date, at revison: $LOCAL
    sleep $INTERVAL;
  fi;
done;)