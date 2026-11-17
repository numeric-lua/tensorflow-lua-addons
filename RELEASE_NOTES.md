# tensorflow-lua-addons 0.2.0

TensorFlow Addons for LuaJIT.

## Added since 0.1.0

- Cohen's kappa metric;
- NovoGrad optimizer;
- CSVLogger callback.

## Included

- activations: gelu, swish, mish, lisht, tanhshrink;
- losses: focal, dice, tversky;
- metrics: F1, F-beta, Matthews correlation, Cohen's kappa, R², Pearson;
- layers: group, instance, and layer normalization;
- optimizers: LAMB and NovoGrad;
- callbacks: TimeStopping, TerminateOnNaN, BackupAndRestore, CSVLogger.

## Requirements

- LuaJIT 2.1
- tensorflow-lua >= 0.8
- libtensorflow 2.16.2/2.18.0 CPU

## Verification

- 12 dependency-free graph tests;
- luacheck clean;
- CI against tensorflow-lua and libtensorflow 2.18.
