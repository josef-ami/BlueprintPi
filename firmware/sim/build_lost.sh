#!/bin/sh
# build_lost.sh [outname] - compile ../ObstacleRound.cpp for the host, unmodified,
# against the stub headers in this folder, exposing the lost-pillar chain.
# -Wall is on: a warning here is a warning the Arduino build would print too.
set -e
cd "$(dirname "$0")"
OUT="${1:-liblost.so}"
g++ -std=gnu++17 -O2 -Wall -fPIC -shared -I. lostexec.cpp -o "$OUT"
echo "built $OUT"
