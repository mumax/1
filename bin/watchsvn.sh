#! /bin/bash
INTERVAL=60




(for (( ;; )); do

  LOCAL=$(svn info | grep "Revision: " | tr -d "Revision: ");
  REMOTE=$(svn stat -uq | grep "Status against revision:" | tr -d "Status against revision:\t");

  if ((( $LOCAL != $REMOTE ))); then
    echo "New SVN update available: revision" $REMOTE
    notify-send "New SVN update available: " $REMOTE
    while ((( $LOCAL != $REMOTE ))); do
      sleep $INTERVAL
      
      LOCAL=$(svn info | grep "Revision: " | tr -d "Revision: ");
      REMOTE=$(svn stat -uq | grep "Status against revision:" | tr -d "Status against revision:\t");

    done;
  else
    echo up to date, at revison: $LOCAL
    sleep $INTERVAL;
  fi;
done;)