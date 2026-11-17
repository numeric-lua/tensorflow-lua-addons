#!/bin/sh
set -eu
ROOT=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
TF_ROOT=${TENSORFLOW_ROOT:-"$ROOT/../tensorflow-lua/third_party/libtensorflow"}
export TENSORFLOW_ROOT="$TF_ROOT"
export LUA_PATH="$ROOT/?.lua;$ROOT/?/init.lua;$ROOT/../tensorflow-lua/?.lua;$ROOT/../tensorflow-lua/?/init.lua;;"
exec luajit "$ROOT/tests/run.lua"
