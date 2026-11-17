#!/bin/sh
set -eu
ROOT=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
if [ -n "${TENSORFLOW_LUA_ROOT:-}" ]; then
  TF_SOURCE="$TENSORFLOW_LUA_ROOT"
elif [ -f "$ROOT/tensorflow-lua/tensorflow/init.lua" ]; then
  TF_SOURCE="$ROOT/tensorflow-lua"
else
  TF_SOURCE="$ROOT/../tensorflow-lua"
fi
TF_ROOT=${TENSORFLOW_ROOT:-"$TF_SOURCE/third_party/libtensorflow"}
export TENSORFLOW_ROOT="$TF_ROOT"
export LUA_PATH="$ROOT/?.lua;$ROOT/?/init.lua;$TF_SOURCE/?.lua;$TF_SOURCE/?/init.lua;;"
exec luajit "$ROOT/tests/run.lua"
