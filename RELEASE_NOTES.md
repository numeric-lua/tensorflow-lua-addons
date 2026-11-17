# tensorflow-lua-addons 0.1.0

The first TensorFlow Addons parity subset for LuaJIT.

## Included

- activations: gelu, swish, mish, lisht, tanhshrink;
- losses: focal, dice, tversky;
- metrics: F1, F-beta, Matthews correlation, R², Pearson;
- layers: group, instance, and layer normalization;
- optimizer: LAMB;
- callbacks: TimeStopping, TerminateOnNaN, BackupAndRestore;
- CI against tensorflow-lua and libtensorflow 2.18.

## Requirements

- LuaJIT 2.1
- tensorflow-lua >= 0.8
- libtensorflow 2.16.2/2.18.0 CPU

This is not complete TensorFlow Addons parity yet. See the README roadmap for
missing spectral/weight normalization, specialized losses, multi-class metrics,
additional optimizers, RNN/text/seq2seq, image augmentation, and logging.

## Install

```sh
luarocks --lua-version=5.1 install tensorflow-lua-addons
```

## Verification

- 9 dependency-free graph tests;
- luacheck clean;
- CI checkout of tensorflow-lua and libtensorflow.
