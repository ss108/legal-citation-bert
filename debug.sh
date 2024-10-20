#!/bin/sh

set -e

if [ "$#" -eq 0 ]
then
  echo "No filename provided. Usage: ./debug.sh <filename>.py"
  exit 1
fi

SCRIPT_NAME=$1

python -m debugpy --listen 0.0.0.0:5678 --wait-for-client $SCRIPT_NAME