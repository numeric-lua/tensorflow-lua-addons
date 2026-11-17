# tensorflow-lua-addons 0.3.0

TensorFlow Addons for LuaJIT.

## Added since 0.2.0

- triplet loss with a configurable margin;
- weight-normalization helper with unit column norms.

## Included

- activations: gelu, swish, mish, lisht, tanhshrink;
- losses: focal, dice, tversky, triplet;
- metrics: F1, F-beta, Matthews correlation, Cohen's kappa, R², Pearson;
- layers: group, instance, layer, and weight normalization;
- optimizers: LAMB and NovoGrad;
- callbacks: TimeStopping, TerminateOnNaN, BackupAndRestore, CSVLogger.

## Requirements

- LuaJIT 2.1
- tensorflow-lua >= 0.8
- libtensorflow 2.16.2/2.18.0 CPU

## Verification

- 14 dependency-free graph tests;
- luacheck clean;
- CI against tensorflow-lua and libtensorflow 2.18.
