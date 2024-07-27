#!/bin/sh

set -e

if [ "$#" -eq 0 ]
then
  echo "No filename provided. Usage: ./debug.sh <filename>.py"
  exit 1
fi

# Extract the script name from the argument
SCRIPT_NAME=$1

# Start the debugpy server and run the provided script
python -m debugpy --listen 0.0.0.0:5678 --wait-for-client $SCRIPT_NAME