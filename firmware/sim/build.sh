#!/bin/sh
# build.sh [outname] - compile ../ObstacleRound.cpp for the host, unmodified,
# against the stub headers in this folder. -Wall is on: a warning here is a
# warning the Arduino build would print too.
set -e
cd "$(dirname "$0")"
OUT="${1:-libexec.so}"
g++ -std=gnu++17 -O2 -Wall -fPIC -shared -I. simexec.cpp -o "$OUT"
echo "built $OUT"
