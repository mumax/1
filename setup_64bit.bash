#! /bin/bash

# Generates bin/setenv

OUTPUT=bin/setenv
echo '#! /bin/bash' > $OUTPUT
echo 'export SIMROOT='$(pwd) >> $OUTPUT
cat src/setenv_64bit.template >> $OUTPUT
chmod u+x $OUTPUT

echo Created $OUTPUT.
echo You can now run this script to start simulation programs,
echo or append \"source $OUTPUT\" to ~/.bashrc