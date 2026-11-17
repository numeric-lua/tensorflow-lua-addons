# tensorflow-lua-addons

Additional activations, losses, metrics, layers, callbacks, and optimizers for
[`tensorflow-lua`](https://github.com/numeric-lua/tensorflow-lua), modeled after
TensorFlow Addons.

This is an early parity subset, not yet a complete TensorFlow Addons port.
See the parity table below for what exists today.

## Install

```sh
luarocks --lua-version=5.1 install tensorflow-lua-addons
```

The package depends on `tensorflow-lua >= 0.8` and requires a local
libtensorflow installation.

## API

```lua
local tf = require("tensorflow")
local tfa = require("tensorflow_addons")

local activation = tfa.activations.gelu(graph, input, "gelu")
local loss = tfa.losses.focal(graph, labels, predictions, { gamma = 2 })
local metric = tfa.metrics.f1_score(graph, predictions, labels)
local layer = tfa.layers.group_normalization(graph, input, 4,
  { input_shape = { 8, 8, 16 } })
```

### Activations

- `gelu` (tanh approximation)
- `swish`
- `mish`
- `lisht`
- `tanhshrink`

### Losses

- `focal`
- `dice`
- `tversky`

### Metrics

- `f1_score`
- `fbeta_score`
- `matthews_corrcoef`
- `r_square`
- `pearson_correlation`

### Layers

- `group_normalization`
- `instance_normalization`
- `layer_normalization`

### Optimizers

- `LAMB`

### Callbacks

- `TimeStopping`
- `TerminateOnNaN`
- `BackupAndRestore`

## Parity roadmap

Still to port from TensorFlow Addons:

- spectral/weight normalization and Poisson/quantization layers;
- triplet/contrastive/GIoU and other specialized losses;
- Cohen kappa and multi-class metric coverage;
- LazyAdam, NovoGrad, Lookahead, and weight-decay optimizer wrappers;
- RNN/text/seq2seq addons;
- image augmentations and preprocessing;
- TQDM and CSV logging callbacks.

## Tests

```sh
TENSORFLOW_ROOT="$PWD/third_party/libtensorflow" ./scripts/test.sh
```

The suite is dependency-free and uses a sibling `tensorflow-lua` checkout by
default.

## License

BSD-3-Clause.
