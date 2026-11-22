# tensorflow-lua-addons

[![❤ Sponsor on GitHub](https://img.shields.io/badge/%E2%9D%A4%20Sponsor-GitHub-ea4aaa?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/Mateus-M-Reis)

Additional activations, losses, metrics, layers, callbacks, and optimizers for
[`tensorflow-lua`](https://github.com/numeric-lua/tensorflow-lua), modeled after
TensorFlow Addons.

This is a 0.10.0 parity subset, not yet a complete TensorFlow Addons port.
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
local robust_loss = tfa.losses.log_cosh(graph, labels, predictions)
local box_loss = tfa.losses.giou(graph, boxes, predicted_boxes)
local metric = tfa.metrics.f1_score(graph, predictions, labels)
local macro_f1 = tfa.metrics.f1_score(graph, predicted_classes, one_hot_labels,
  { average = "macro", num_classes = 3 })
local iou = tfa.metrics.mean_iou(graph, predicted_classes, labels,
  { num_classes = 3 })
local kappa = tfa.metrics.cohen_kappa(graph, confusion_matrix,
  { weights = "quadratic" })
local layer = tfa.layers.group_normalization(graph, input, 4,
  { input_shape = { 8, 8, 16 } })
local maxout = tfa.layers.maxout(graph, input, 2)
local blurred = tfa.image.gaussian_filter2d(graph, image,
  { filter_shape = { 3, 3 }, sigma = 1 })
local lazy = tfa.optimizers.LazyAdam(graph, loss, { weight },
  { learning_rate = 0.001 })
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
- `log_cosh`

`giou` accepts boxes shaped `[..., 4]` in `[x1, y1, x2, y2]` order. It pairs
the leading dimensions one-vs-one and returns the mean `1 - GIoU` loss. Use
`options.name`, `options.dtype`, and `options.epsilon` (default `1e-7`) to
configure the graph namespace, numeric type, and zero-area denominator guard.
`tfa.losses.giou_loss` is provided as a TensorFlow Addons-style alias.

`log_cosh` computes the numerically stable log-cosh loss
`|error| + softplus(-2|error|) - log(2)` and accepts
`options.reduction = "none" | "mean" | "sum"` (default `"mean"`, also spelled
`"sum_over_batch_size"`). `tfa.losses.LogCosh` is an alias.

Loss roadmap: `focal`, `dice`, `tversky`, `triplet`, `contrastive`, `giou`, and
`log_cosh` are available; further specialized losses remain on the parity
roadmap.

### Metrics

- `f1_score` (binary threshold or multiclass averages)
- `fbeta_score`
- `matthews_corrcoef`
- `cohen_kappa` (binary or multiclass weighted)
- `r_square`
- `pearson_correlation`
- `mean_iou`
- `multiclass_f1` (legacy macro-F1 helper)

`f1_score(graph, predictions, labels, options)` keeps the binary threshold API
by default: scalar predictions are compared with `options.threshold` (default
`0.5`). Set `options.average` to `"macro"`, `"micro"`, or `"weighted"` to use
multiclass averaging. Multiclass inputs may be integer class ids or one-hot /
score rows; `options.num_classes` sets the class count, and when it is omitted
the count is inferred from one-hot widths or the largest class id. Use
`options.predictions_one_hot` and `options.labels_one_hot` to override the
integer-versus-one-hot inference. `"weighted"` weights each class by its true
label support, while `"micro"` computes a single global F1 from all classes.

`cohen_kappa(graph, predictions, labels, options)` also keeps the binary
threshold API. For multiclass kappa, pass integer class ids or one-hot rows and
set `options.num_classes` when the class count cannot be inferred. Weighted
kappa uses `options.weights = "none" | "linear" | "quadratic"` (default
`"none"`); `options.weightage` is accepted as an alias. An already-computed
confusion matrix can be passed as the `predictions` argument:

```lua
local kappa = tfa.metrics.cohen_kappa(graph, confusion_matrix,
  { weights = "quadratic" })
```

`mean_iou(graph, predictions, labels, num_classes, options)` accepts integer
class ids or one-hot rows and returns the mean intersection-over-union over
classes whose union is present in the batch. `num_classes` may also be supplied
as `options.num_classes`; pass one-hot inputs through
`options.predictions_one_hot` / `options.labels_one_hot` when inference is
ambiguous. `tfa.metrics.MeanIoU` is an alias.

Classification metrics accept `options.name` for the graph namespace.

### Layers

- `group_normalization`
- `instance_normalization`
- `layer_normalization`
- `weight_normalization`
- `spectral_normalization`
- `maxout`

### Layers extras

- `poisson` (deterministic rate or noise inverse-CDF);
- `quantization` (fake-quant).

`maxout(graph, input, num_units, options)` groups the selected axis
(`options.axis`, default last) into `num_units` groups and returns the group
maximum. Pass `options.input_shape` when using a non-last axis.
`tfa.layers.Maxout` is an alias.

### Optimizers

- `LAMB`
- `NovoGrad`
- `LazyAdam`
- `Lookahead` (wrapper)
- `WeightDecay` (wrapper)

`LazyAdam(graph, loss, variables, options)` follows TensorFlow Addons'
sparse-friendly Adam semantics: moment accumulators and variables are only
updated where the gradient is non-zero, while the beta-power schedule advances
on every step. It accepts the same learning-rate, beta, epsilon, and
weight-decay options as `LAMB`.

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

### Image

- `flip_left_right`
- `flip_up_down`
- `adjust_brightness`
- `resize_nearest`
- `gaussian_filter2d`

`gaussian_filter2d(graph, image, options)` blurs rank-4 NHWC images (or rank-3
HWC images with `options.rank = 3`) using an odd-sized Gaussian kernel. It
accepts `filter_shape` (default `{ 3, 3 }`), `sigma` (number or
`{ sigma_y, sigma_x }`), and `padding` (`"REFLECT"` by default, plus
`"SYMMETRIC"`, `"CONSTANT"`, `"SAME"`, and `"VALID"`).

### Callbacks

- `TimeStopping`
- `TerminateOnNaN`
- `BackupAndRestore`
- `CSVLogger`

## Parity status

| Area | Implemented | Remaining TensorFlow Addons work |
| --- | --- | --- |
| Activations | `gelu`, `swish`, `mish`, `lisht`, `tanhshrink` | additional activation wrappers |
| Losses | `focal`, `dice`, `tversky`, `triplet`, `contrastive`, `giou`, `log_cosh` | hinge, KL divergence, N-pairs, pinball/quantile, sigmoid focal, semi-hard triplet |
| Metrics | `f1_score`, `fbeta_score`, `matthews_corrcoef`, `cohen_kappa`, `r_square`, `pearson_correlation`, `mean_iou`, `multiclass_f1` | multilabel confusion matrix, additional detection/segmentation metrics |
| Normalization layers | `group`, `instance`, `layer`, `weight`, `spectral` | filter-response normalization and further normalization variants |
| Other layers | `poisson`, `quantization`, `maxout` | ESN, multi-head attention, noisy dense, stochastic depth |
| Optimizers | `LAMB`, `NovoGrad`, `LazyAdam`, `Lookahead`, `WeightDecay` | averaged/conditional-gradient and further optimizer wrappers |
| Image | horizontal/vertical flips, brightness, nearest resize, `gaussian_filter2d` | random color augmentation, warps/transforms, bilinear/area resize, median/Sobel filters |
| Callbacks | `TimeStopping`, `TerminateOnNaN`, `BackupAndRestore`, `CSVLogger` | `TQDMProgressBar` |
| RNN/text/seq2seq | none yet | attention, text, and sequence-to-sequence addons |

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
