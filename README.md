# tensorflow-lua-addons

[![❤ Sponsor on GitHub](https://img.shields.io/badge/%E2%9D%A4%20Sponsor-GitHub-ea4aaa?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/Mateus-M-Reis)

Additional activations, losses, metrics, layers, callbacks, and optimizers for
[`tensorflow-lua`](https://github.com/numeric-lua/tensorflow-lua), modeled after
TensorFlow Addons.

This is a 0.6.0 parity subset, not yet a complete TensorFlow Addons port.
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
local box_loss = tfa.losses.giou(graph, boxes, predicted_boxes)
local metric = tfa.metrics.f1_score(graph, predictions, labels)
local layer = tfa.layers.group_normalization(graph, input, 4,
  { input_shape = { 8, 8, 16 } })
local optimizer = tfa.optimizers.Lookahead(tfa.optimizers.LAMB,
  { sync_period = 5, slow_step_size = 0.5, learning_rate = 0.001 })
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
- `triplet`
- `contrastive`
- `giou`

`giou` accepts boxes shaped `[..., 4]` in `[x1, y1, x2, y2]` order. It pairs
the leading dimensions one-vs-one and returns the mean `1 - GIoU` loss. Use
`options.name`, `options.dtype`, and `options.epsilon` (default `1e-7`) to
configure the graph namespace, numeric type, and zero-area denominator guard.
`tfa.losses.giou_loss` is provided as a TensorFlow Addons-style alias.

Loss roadmap: `focal`, `dice`, `tversky`, `triplet`, `contrastive`, and `giou`
are available; further specialized losses remain on the parity roadmap.

### Metrics

- `f1_score`
- `fbeta_score`
- `matthews_corrcoef`
- `cohen_kappa`
- `r_square`
- `pearson_correlation`

### Layers

- `group_normalization`
- `instance_normalization`
- `layer_normalization`
- `weight_normalization`
- `spectral_normalization`

### Optimizers

- `LAMB`
- `NovoGrad`
- `Lookahead` (wrapper)

`Lookahead(base_optimizer, options)` wraps a core optimizer function such as
`tfa.optimizers.LAMB` or `tfa.optimizers.NovoGrad`. It keeps a slow copy of
every variable and updates the fast weights through the base optimizer. Every
`sync_period` steps (default `5`) the fast weights are blended toward the slow
weights by `slow_step_size` (default `0.5`) and both copies are synchronized.

```lua
local lookahead = tfa.optimizers.Lookahead(tfa.optimizers.LAMB,
  { sync_period = 5, slow_step_size = 0.5, learning_rate = 0.01 })
local targets, gradients, state = lookahead(graph, loss, { weight })
session:run({}, {}, { weight.initializer, unpack(state.initializers) })
session:run(feeds, {}, targets)
```

Options other than `sync_period`, `slow_step_size`, `name`, and `base_name` are
forwarded to the base optimizer; call-time options override construction-time
options. The returned state includes the usual `initializers` and `variables`,
plus `slow_weights` (parallel to the variable list), `step`, `base_state`, and
the resolved `sync_period`/`slow_step_size`.

### Callbacks

- `TimeStopping`
- `TerminateOnNaN`
- `BackupAndRestore`
- `CSVLogger`

## Parity roadmap

Still to port from TensorFlow Addons:

- Poisson/quantization layers;
- further specialized losses beyond the implemented set;
- Cohen kappa and multi-class metric coverage;
- LazyAdam and weight-decay optimizer wrappers;
- RNN/text/seq2seq addons;
- image augmentations and preprocessing;
- TQDM and CSV logging callbacks.

## Tests

```sh
TENSORFLOW_ROOT="../tensorflow-lua/third_party/libtensorflow" ./scripts/test.sh
```

The suite is dependency-free and uses a sibling `tensorflow-lua` checkout by
default.

## Support

Support is optional and never unlocks features. See the organization support
page: <https://github.com/numeric-lua/.github/blob/main/SUPPORT.md>.

## License

BSD-3-Clause.
