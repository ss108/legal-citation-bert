#!/bin/sh
set -e

echo "## Installing Python dependencies ##" && pdm install --global --project .

# keep container alive until explicitly shut down
tail -f /dev/null
