#!/bin/bash
# Place this file in the install folder of the application so the shell is automatically sourced
set -e
source ./setup.bash
# Executed arguments in the sourced shell
exec "$@"
