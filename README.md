# tensorflow-lua-addons

[![❤ Sponsor on GitHub](https://img.shields.io/badge/%E2%9D%A4%20Sponsor-GitHub-ea4aaa?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/Mateus-M-Reis)

Additional activations, losses, metrics, layers, callbacks, and optimizers for
[`tensorflow-lua`](https://github.com/numeric-lua/tensorflow-lua), modeled after
TensorFlow Addons.

This is a 0.17.0 parity subset, not yet a complete TensorFlow Addons port.
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
local periodic = tfa.activations.snake(graph, input, 1)
local loss = tfa.losses.focal(graph, labels, predictions, { gamma = 2 })
local robust_loss = tfa.losses.log_cosh(graph, labels, predictions)
local quantile_loss = tfa.losses.pinball(graph, labels, predictions,
  { tau = 0.1 })
local focal_ce = tfa.losses.sigmoid_focal_crossentropy(graph, labels,
  predictions, { alpha = 0.25, gamma = 2 })
local box_loss = tfa.losses.giou(graph, boxes, predicted_boxes)
local hinge = tfa.losses.hinge(graph, signed_labels, predictions,
  { margin = 1 })
local kl = tfa.losses.kl_divergence(graph, label_distribution,
  predicted_distribution, { reduction = "sum_over_batch_size" })
local lifted = tfa.losses.lifted(graph, class_ids, embeddings)
local kappa_loss = tfa.losses.weighted_kappa(graph, one_hot_labels, scores,
  { num_classes = 4 })
local sparse_loss = tfa.losses.sparsemax(graph, logits, one_hot_labels)
local metric = tfa.metrics.f1_score(graph, predictions, labels)
local precision = tfa.metrics.precision(graph, predicted_classes, labels,
  { average = "macro", num_classes = 3 })
local recall = tfa.metrics.recall(graph, predicted_classes, labels,
  { average = "macro", num_classes = 3 })
local macro_f1 = tfa.metrics.f1_score(graph, predicted_classes, one_hot_labels,
  { average = "macro", num_classes = 3 })
local iou = tfa.metrics.mean_iou(graph, predicted_classes, labels,
  { num_classes = 3 })
local kappa = tfa.metrics.cohen_kappa(graph, confusion_matrix,
  { weights = "quadratic" })
local mlcm = tfa.metrics.multilabel_confusion_matrix(graph, predictions, labels,
  3)
local hamming = tfa.metrics.hamming_loss(graph, predictions, labels,
  { mode = "multiclass", threshold = 0.5 })
local auc = tfa.metrics.auc(graph, predictions, labels, { curve = "ROC" })
local sensitivity = tfa.metrics.sensitivity_at_specificity(graph, predictions,
  labels, 0.9)
local specificity = tfa.metrics.specificity_at_sensitivity(graph, predictions,
  labels, 0.9)
local precision_at_recall = tfa.metrics.precision_at_recall(graph, predictions,
  labels, 0.8)
local recall_at_precision = tfa.metrics.recall_at_precision(graph, predictions,
  labels, 0.8)
local class_precision = tfa.metrics.precision(graph, scores, one_hot_labels,
  { class_id = 1 })
local geometric = tfa.metrics.geometric_mean(graph, values)
local harmonic = tfa.metrics.harmonic_mean(graph, values)
local npairs = tfa.losses.npairs_loss(graph, class_ids, similarity_matrix)
local semihard = tfa.losses.triplet_semihard_loss(graph, class_ids, embeddings,
  { margin = 1, distance_metric = "L2" })
local frn = tfa.layers.filter_response_normalization(graph, input,
  { input_shape = { 8, 8, 8, 16 } })
local layer = tfa.layers.group_normalization(graph, input, 4,
  { input_shape = { 8, 8, 16 } })
local maxout = tfa.layers.maxout(graph, input, 2)
local ball = tfa.layers.poincare_normalize(graph, input, { axis = 1 })
local thresholded = tfa.layers.tlu(graph, input, { input_shape = { 8, 16 } })
local crossed = tfa.layers.polynomial_crossing(graph, x0, x,
  { input_shape = { 8, 16 } })
local sparse = tfa.layers.sparsemax(graph, input, { input_shape = { 8, 16 } })
local adaptive = tfa.layers.adaptive_average_pooling_2d(graph, image, 4,
  { input_shape = { 1, 8, 8, 16 } })
local adaptive_1d = tfa.layers.adaptive_average_pooling_1d(graph, sequence, 4,
  { input_shape = { 1, 16, 8 } })
local adaptive_3d = tfa.layers.adaptive_average_pooling_3d(graph, volume, 2,
  { input_shape = { 1, 4, 4, 4, 8 } })
local vlad = tfa.layers.netvlad(graph, frames, 8,
  { input_shape = { 1, 16, 32 } })
local unpooled = tfa.layers.max_unpooling_2d(graph, pooled, mask,
  { pool_size = { 2, 2 }, strides = { 2, 2 } })
local snake_layer = tfa.layers.snake(graph, input, { frequency = 1 })
local gelu_layer = tfa.layers.gelu(graph, input, { name = "gelu_layer" })
local pyramid = tfa.layers.spatial_pyramid_pooling_2d(graph, image, { 1, 2, 3 },
  { input_shape = { 1, 8, 8, 16 } })
local reservoir = tfa.layers.esn(graph, sequence, 32,
  { input_shape = { 1, 16, 8 }, return_sequences = true })
local attention = tfa.layers.multi_head_attention(graph, sequence, sequence,
  sequence, { query_shape = { 1, 16, 8 }, num_heads = 4, key_dim = 8 })
local rms = tfa.layers.rms_normalization(graph, input)
local standardized = tfa.layers.standardize(graph, input)
local noisy = tfa.layers.noisy_dense(graph, input, 32,
  { input_shape = { 8, 16 } })
local depth = tfa.layers.stochastic_depth(graph, shortcut, residual,
  { survival_probability = 0.5, training = true })
local blurred = tfa.image.gaussian_filter2d(graph, image,
  { filter_shape = { 3, 3 }, sigma = 1 })
local median = tfa.image.median_filter2d(graph, image,
  { filter_shape = { 3, 3 }, padding = "REFLECT" })
local mean = tfa.image.mean_filter2d(graph, image,
  { filter_shape = { 3, 3 }, padding = "REFLECT" })
local edges = tfa.image.sobel_edges(graph, image,
  { input_shape = { 1, 8, 8, 3 } })
local shifted = tfa.image.translate(graph, image, { 2, 0 },
  { interpolation = "bilinear" })
local rotated = tfa.image.rotate(graph, image, math.pi / 4,
  { interpolation = "bilinear" })
local recolored = tfa.image.random_hsv_in_yiq(graph, image,
  { max_delta_hue = 0.1, lower_saturation = 0.8, upper_saturation = 1.2 })
local muted = tfa.image.adjust_saturation(graph, image, 0.5)
local hued = tfa.image.adjust_hue(graph, image, 0.25)
local hsv = tfa.image.adjust_hsv_in_yiq(graph, image, 0.1, 1.1, 0.9)
local gamma = tfa.image.adjust_gamma(graph, image, 0.8, 1)
local contrasted = tfa.image.adjust_contrast(graph, image, 1.2)
local hsv_image = tfa.image.rgb_to_hsv(graph, image)
local gray_image = tfa.image.rgb_to_grayscale(graph, image)
local warped = tfa.image.dense_image_warp(graph, image, flow)
local transformed = tfa.image.transform(graph, image,
  { 1, 0, 0, 0, 1, 0, 0, 0 })
local composed = tfa.image.compose_transforms(graph,
  { { 1, 0, 1, 0, 1, 0, 0, 0 }, { 1, 0, 1, 0, 1, 0, 0, 0 } })
local angle_rows = tfa.image.angles_to_projective_transforms(graph,
  math.pi / 4, 8, 8)
local translation_rows = tfa.image.translations_to_projective_transforms(
  graph, { 2, 1 })
local transform_matrix = tfa.image.flat_transforms_to_matrices(graph,
  translation_rows)
local flat_rows = tfa.image.matrices_to_flat_transforms(graph,
  transform_matrix)
local sheared_x = tfa.image.shear_x(graph, image, 0.25)
local sheared_y = tfa.image.shear_y(graph, image, 0.25)
local random_bright = tfa.image.random_brightness(graph, image, 0.1)
local random_contrast = tfa.image.random_contrast(graph, image, 0.8, 1.2)
local random_saturation = tfa.image.random_saturation(graph, image, 0.8, 1.2)
local random_hue = tfa.image.random_hue(graph, image, 0.05)
local random_crop = tfa.image.random_crop(graph, image, { 4, 4 })
local cut = tfa.image.cutout(graph, image, { 4, 4 }, { 8, 8 })
local random_cut = tfa.image.random_cutout(graph, image, { 4, 4 })
local small = tfa.image.bilinear_resize(graph, image, 4, 4)
local pooled = tfa.image.area_resize(graph, image, 4, 4)
local bicubic = tfa.image.resize_bicubic(graph, image, 4, 4)
local crops = tfa.image.crop_and_resize(graph, image, { { 0, 0, 1, 1 } }, { 0 },
  { 4, 4 })
local padded_box = tfa.image.pad_to_bounding_box(graph, image, 2, 2, 16, 16,
  { input_shape = { 1, 8, 8, 3 } })
local cropped_box = tfa.image.crop_to_bounding_box(graph, image, 2, 2, 4, 4,
  { input_shape = { 1, 8, 8, 3 } })
local fitted = tfa.image.resize_with_pad(graph, image, 12, 12,
  { input_shape = { 1, 8, 8, 3 } })
local center = tfa.image.central_crop(graph, image, 0.5,
  { input_shape = { 1, 8, 8, 3 } })
local sharpened = tfa.image.sharpness(graph, image, 0.5)
local mixed = tfa.image.blend(graph, image, other_image, 0.5)
local conditional = tfa.optimizers.ConditionalGradient(graph, loss, { weight },
  { learning_rate = 0.01, lambda_ = 0.01 })
local yogi = tfa.optimizers.Yogi(graph, loss, { weight },
  { learning_rate = 0.01 })
local adamw = tfa.optimizers.AdamW(graph, loss, { weight },
  { learning_rate = 0.001, weight_decay = 0.004 })
local sgdw = tfa.optimizers.SGDW(graph, loss, { weight },
  { learning_rate = 0.01, weight_decay = 0.004 })
local moving = tfa.optimizers.MovingAverage(tfa.optimizers.LAMB,
  { average_decay = 0.99 })
local lazy = tfa.optimizers.LazyAdam(graph, loss, { weight },
  { learning_rate = 0.001 })
local optimizer = tfa.optimizers.Lookahead(tfa.optimizers.LAMB,
  { sync_period = 5, slow_step_size = 0.5, learning_rate = 0.001 })
local progress = tfa.callbacks.TQDMProgressBar()
```

### Activations

- `gelu` (tanh approximation)
- `swish`
- `mish`
- `lisht`
- `tanhshrink`
- `snake`

`snake(graph, input, frequency, name, dtype)` implements the periodic Snake
activation `x + (1 - cos(2 * frequency * x)) / (2 * frequency)`. Frequency may
be a Lua number or a scalar graph tensor.

### Losses

- `focal`
- `dice`
- `tversky`
- `triplet`
- `contrastive`
- `giou`
- `log_cosh`
- `hinge`
- `kl_divergence`
- `pinball`
- `sigmoid_focal_crossentropy`
- `npairs_loss`
- `npairs_multilabel_loss`
- `triplet_semihard_loss`
- `lifted`
- `weighted_kappa`
- `sparsemax`

`giou` accepts boxes shaped `[..., 4]` in `[x1, y1, x2, y2]` order. It pairs
the leading dimensions one-vs-one and returns the mean `1 - GIoU` loss. Use
`options.name`, `options.dtype`, and `options.epsilon` (default `1e-7`) to
configure the graph namespace, numeric type, and zero-area denominator guard.
`tfa.losses.giou_loss` is provided as a TensorFlow Addons-style alias.

`log_cosh` computes the numerically stable log-cosh loss
`|error| + softplus(-2|error|) - log(2)` and accepts
`options.reduction = "none" | "mean" | "sum"` (default `"mean"`, also spelled
`"sum_over_batch_size"`). `tfa.losses.LogCosh` is an alias.

`hinge(graph, labels, predictions, options)` implements binary hinge loss for
labels in `{-1, 1}`: each element is `maximum(margin - labels * predictions, 0)`.
`options.margin` defaults to `1`; reductions are `"none"`, `"mean"`, or `"sum"`
(default `"sum_over_batch_size"`, which is equivalent to `"mean"`).
`tfa.losses.Hinge` and `tfa.losses.hinge_loss` are aliases.

`kl_divergence(graph, labels, predictions, options)` follows Keras'
`kl_divergence`: both inputs are probability distributions along the last axis,
and each example contributes `sum(labels * log(labels / predictions))`. Zero
labels contribute zero, and predictions are floored by `options.epsilon`
(default `1e-7`) to keep the result finite. Reductions are `"none"`, `"mean"`,
or `"sum"` (default `"sum_over_batch_size"`). `tfa.losses.KLDivergence` and
`tfa.losses.kl_divergence_loss` are aliases.

`pinball(graph, labels, predictions, options)` implements the quantile loss
`maximum(tau * (labels - predictions), (tau - 1) * (labels - predictions))`.
`options.tau` defaults to `0.5` and must be in `[0, 1]`; reductions are
`"none"`, `"mean"` (default), or `"sum"`. `tfa.losses.PinballLoss` and
`tfa.losses.pinball_loss` are aliases.

`sigmoid_focal_crossentropy(graph, labels, predictions, options)` implements
the RetinaNet sigmoid focal loss with `options.alpha` (default `0.25`),
`options.gamma` (default `2`), `options.from_logits` (default `false`), and
`options.reduction` (`"none"` by default, matching the TensorFlow Addons
function; `"mean"` and `"sum"` are also accepted).
`tfa.losses.SigmoidFocalCrossEntropy` is an alias.

`npairs_loss(graph, labels, similarity, options)` follows
`tfa.losses.npairs_loss`: `labels` are 1-D class ids and `similarity` is the
`[batch_size, batch_size]` matrix of pairwise logits. Rows with equal labels
are normalized into a soft target distribution and the result is the mean
softmax cross entropy. `npairs_multilabel_loss` accepts binary indicator rows
instead, using the count of shared labels as the adjacency matrix.
`tfa.losses.NpairsLoss` and `tfa.losses.NpairsMultilabelLoss` are aliases.

`triplet_semihard_loss(graph, labels, embeddings, options)` implements
`tfa.losses.triplet_semihard_loss` with in-batch semi-hard negative mining.
`options.margin` defaults to `1` and `options.distance_metric` accepts `"L2"`
(default), `"squared-L2"`, or `"angular"`. `tfa.losses.TripletSemiHardLoss` is
an alias.

`lifted(graph, labels, embeddings, options)` computes the lifted structured
loss over in-batch class ids and embedding vectors, following
`tfa.losses.lifted_struct_loss`; `options.margin` defaults to `1`.
`tfa.losses.LiftedStructLoss` and `tfa.losses.lifted_struct_loss` are aliases.

`weighted_kappa(graph, labels, predictions, options)` implements
`tfa.losses.WeightedKappaLoss` for ordinal multi-class predictions. It requires
`options.num_classes`, accepts `options.weightage = "linear" | "quadratic"`
(default `"quadratic"`), and uses `options.epsilon` (default `1e-6`).
`tfa.losses.WeightedKappaLoss` is an alias.

`sparsemax(graph, logits, labels, options)` implements the Sparsemax loss from
logits, following `tfa.losses.SparsemaxLoss`. Reductions are `"none"`, `"mean"`,
or `"sum"` (default `"sum_over_batch_size"`). `tfa.losses.SparsemaxLoss`,
`tfa.losses.sparsemax_loss`, and `tfa.losses.sparsemax_loss_from_logits` are
aliases.

Loss roadmap: `focal`, `dice`, `tversky`, `triplet`, `contrastive`, `giou`,
`log_cosh`, `hinge`, `kl_divergence`, `pinball`,
`sigmoid_focal_crossentropy`, `npairs_loss`, `npairs_multilabel_loss`,
`triplet_semihard_loss`, `lifted`, `weighted_kappa`, and `sparsemax` are
available; hard-triplet mining and further specialized losses remain on the
parity roadmap.

### Metrics

- `f1_score` (binary threshold or multiclass averages)
- `precision` (binary threshold or multiclass averages)
- `recall` (binary threshold or multiclass averages)
- `fbeta_score`
- `matthews_corrcoef`
- `cohen_kappa` (binary or multiclass weighted)
- `r_square`
- `pearson_correlation`
- `mean_iou`
- `multilabel_confusion_matrix`
- `hamming_loss`
- `hamming_distance`
- `auc` (binary ROC or precision-recall)
- `sensitivity_at_specificity`
- `specificity_at_sensitivity`
- `precision_at_recall`
- `recall_at_precision`
- `geometric_mean`
- `harmonic_mean`
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

`precision`, `recall`, and `f1_score` also accept `options.class_id` to compute
the binary metric for one class selected from the last axis. Predictions and
labels should be rank-2 or higher with matching shapes; scores are compared
with `options.threshold`.

`precision(graph, predictions, labels, options)` and
`recall(graph, predictions, labels, options)` share that binary threshold and
multiclass averaging API. Binary inputs use `options.threshold` (default
`0.5`); multiclass inputs accept integer class ids or one-hot rows and
`options.average = "macro"` (default), `"micro"`, or `"weighted"`.
`tfa.metrics.Precision` and `tfa.metrics.Recall` are aliases. Both accept
`options.class_id` for one-vs-rest binary precision/recall.

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

`multilabel_confusion_matrix(graph, predictions, labels, num_classes, options)`
builds a `[num_classes, 2, 2]` tensor whose cells are
`[[TN, FP], [FN, TP]]` for each binary class. Inputs are binary indicator rows;
set `options.threshold` (with optional `options.from_logits`) to threshold score
rows first. `tfa.metrics.MultiLabelConfusionMatrix` is an alias.

`hamming_loss(graph, predictions, labels, options)` follows
`tfa.metrics.HammingLoss`. `options.mode` is `"multiclass"` (default) or
`"multilabel"`. With `options.threshold` set, predictions above it are active;
without a threshold the per-row maximum is selected and all-zero rows stay
inactive. `hamming_distance(graph, actuals, predictions, options)` returns the
fraction of differing element positions. `tfa.metrics.HammingLoss` is an alias.

`auc(graph, predictions, labels, options)` approximates binary ROC or
precision-recall area under the curve with a threshold sweep, following
`tf.keras.metrics.AUC`. `options.num_thresholds` defaults to `200`; set
`options.thresholds` to supply the interior thresholds instead. Use
`options.curve = "ROC"` (default) or `"PR"`, `options.summation_method =
"interpolation"` (default), `"minoring"`, or `"majoring"`, and
`options.from_logits = true` for logits. Predictions are flattened, so dense
binary and multilabel tensors are treated as one label dimension.

`sensitivity_at_specificity(graph, predictions, labels, specificity, options)`
returns the best sensitivity among thresholded operating points whose
specificity is at least the requested value. `specificity_at_sensitivity(graph,
predictions, labels, sensitivity, options)` is the symmetric metric. Both
accept `options.num_thresholds` (default `200`) and `options.from_logits`.
`tfa.metrics.SensitivityAtSpecificity` and
`tfa.metrics.SpecificityAtSensitivity` are aliases.

`precision_at_recall(graph, predictions, labels, recall, options)` returns the
best precision among operating points with recall at least the requested value;
`recall_at_precision(graph, predictions, labels, precision, options)` is the
symmetric metric. Both accept `options.num_thresholds` (default `200`) and
`options.from_logits`. `tfa.metrics.PrecisionAtRecall` and
`tfa.metrics.RecallAtPrecision` are aliases.

`geometric_mean(graph, values, options)` and
`harmonic_mean(graph, values, options)` compute the geometric and harmonic
means over all tensor elements, following `tfa.metrics.GeometricMean` and
`tfa.metrics.HarmonicMean`. `tfa.metrics.GeometricMean` and
`tfa.metrics.HarmonicMean` are aliases.

Classification metrics accept `options.name` for the graph namespace.

### Layers

- `group_normalization`
- `instance_normalization`
- `layer_normalization`
- `weight_normalization`
- `spectral_normalization`
- `filter_response_normalization`
- `rms_normalization`
- `unit_normalization`
- `standardize`
- `local_response_normalization`
- `maxout`
- `poincare_normalize`
- `tlu`
- `polynomial_crossing`
- `sparsemax`
- `adaptive_average_pooling_1d`
- `adaptive_max_pooling_1d`
- `adaptive_average_pooling_2d`
- `adaptive_max_pooling_2d`
- `adaptive_average_pooling_3d`
- `adaptive_max_pooling_3d`
- `netvlad`
- `max_unpooling_2d`
- `snake`
- `gelu`
- `spatial_pyramid_pooling_2d`
- `esn`
- `multi_head_attention`
- `noisy_dense`
- `stochastic_depth`

### Layers extras

- `poisson` (deterministic rate or noise inverse-CDF);
- `quantization` (fake-quant).

`filter_response_normalization(graph, input, options)` follows
`tfa.layers.FilterResponseNormalization` for NHWC tensors: each channel is
normalized by the root mean square over spatial axes `{ 1, 2 }`, then scaled
and shifted by per-channel `gamma` and `beta`. It requires
`options.input_shape`, accepts `options.epsilon` (default `1e-6`), and supports
`options.learned_epsilon = true` for the trainable positive epsilon term.
`tfa.layers.FilterResponseNormalization` is an alias.

`maxout(graph, input, num_units, options)` groups the selected axis
(`options.axis`, default last) into `num_units` groups and returns the group
maximum. Pass `options.input_shape` when using a non-last axis.
`tfa.layers.Maxout` is an alias.

`poincare_normalize(graph, input, options)` follows
`tfa.layers.PoincareNormalize`: each slice along `options.axis` (default `1`)
is projected into the Poincare ball with norm at most `1 - epsilon`
(`options.epsilon`, default `1e-5`). Slices already inside the ball are
unchanged. The result table exposes `output`, `axis`, and `epsilon`.
`tfa.layers.PoincareNormalize` is an alias.

`tlu(graph, input, options)` follows `tfa.layers.TLU`: each non-batch
coordinate has a learnable threshold `tau`, and `options.affine = true` adds a
learnable slope `alpha` to the threshold. Use `options.tau`/`options.alpha` for
custom initial values. The result exposes `output`, `tau`, `alpha`, `variables`,
and `initializers`. `tfa.layers.TLU` is an alias.

`polynomial_crossing(graph, x0, x, options)` follows
`tfa.layers.PolynomialCrossing`, returning
`x0 * (W x + diag_scale * x) + bias + x`. Set `options.projection_dim` for a
low-rank factorization, `options.diag_scale`, and `options.use_bias`. The
result exposes `output`, the kernel variables, `bias`, `variables`, and
`initializers`. `tfa.layers.PolynomialCrossing` is an alias.

`sparsemax(graph, input, options)` follows `tfa.layers.Sparsemax`, projecting
logits onto the probability simplex along `options.axis` (default last). The
result exposes `output` and the resolved `axis`. `tfa.layers.Sparsemax` is an
alias.

`adaptive_average_pooling_2d(graph, input, output_size, options)` and
`adaptive_max_pooling_2d(graph, input, output_size, options)` follow TensorFlow
Addons' adaptive 2D pooling layers. They accept int or `{rows, cols}` output
sizes and `options.data_format = "channels_last"` (default) or
`"channels_first"`. Input height and width must be divisible by the requested
bins. `tfa.layers.AdaptiveAveragePooling2D` and
`tfa.layers.AdaptiveMaxPooling2D` are aliases.

`adaptive_average_pooling_1d(graph, input, output_size, options)` and
`adaptive_max_pooling_1d(graph, input, output_size, options)` provide the
corresponding rank-3 adaptive 1D pooling layers. Input length must be divisible
by `output_size`; `options.data_format` supports channels-last and
channels-first. `tfa.layers.AdaptiveAveragePooling1D` and
`tfa.layers.AdaptiveMaxPooling1D` are aliases.

`adaptive_average_pooling_3d(graph, input, output_size, options)` and
`adaptive_max_pooling_3d(graph, input, output_size, options)` provide rank-5
adaptive 3D pooling. Spatial dimensions must be divisible by the requested
`{depth, rows, cols}` bins. `tfa.layers.AdaptiveAveragePooling3D` and
`tfa.layers.AdaptiveMaxPooling3D` are aliases.

`netvlad(graph, input, num_clusters, options)` follows `tfa.layers.NetVLAD`,
aggregating `[batch, time, feature]` frames into a normalized
`[batch, feature * clusters]` descriptor. It requires `options.input_shape` and
returns the kernel, bias, and cluster-center variables.

`max_unpooling_2d(graph, updates, mask, options)` follows
`tfa.layers.MaxUnpooling2D`, scattering pooled values back to their argmax
positions. It accepts `options.pool_size`, `options.strides`, and
`options.padding = "SAME" | "VALID"`.

`snake(graph, input, options)` is the trainable-frequency Snake layer wrapper;
it accepts `options.frequency` as the initial positive frequency and returns the
frequency variable with the output. `tfa.layers.Snake` is an alias.

`gelu(graph, input, options)` exposes the tanh-approximation GELU activation as
a layer. `tfa.layers.GELU` is an alias.

`spatial_pyramid_pooling_2d(graph, input, bins, options)` follows
`tfa.layers.SpatialPyramidPooling2D`, cropping each bin to divisible spatial
dimensions, applying adaptive average pooling, flattening the bins, and
concatenating them along the bin axis. It supports channels-last and
channels-first inputs. `tfa.layers.SpatialPyramidPooling2D` is an alias.

`esn(graph, input, units, options)` follows `tfa.layers.ESN` for a fixed
reservoir over statically-sized `[batch, time, feature]` inputs. The recurrent
matrix is connectivity-masked and scaled by its Frobenius norm, an upper bound
on the spectral radius. Options include `connectivity`, `leaky`,
`spectral_radius`, `use_bias`, `activation = "tanh" | "linear"`, and
`return_sequences`. `tfa.layers.ESN` is an alias.

`multi_head_attention(graph, query, key, value, options)` implements scaled
dot-product multi-head attention for rank-3 tensors. It accepts
`query_shape`/`key_shape`/`value_shape`, `num_heads`, `key_dim`, `value_dim`,
and `use_bias`, returning the output, attention scores, variables, and
initializers. `tfa.layers.MultiHeadAttention` is an alias.

`rms_normalization`, `unit_normalization`, and `standardize` provide last-axis
(or `options.axis`) normalization variants. `local_response_normalization`
wraps TensorFlow's LRN op with configurable `depth_radius`, `bias`, `alpha`,
and `beta`. Their capitalized aliases are available.

`noisy_dense(graph, input, units, options)` follows `tfa.layers.NoisyDense`:
it learns mean and sigma kernel/bias variables and multiplies the sigma terms
by factorised Gaussian noise (or independent noise with
`options.use_factorised = false`). It requires `options.input_shape`, accepts
`options.sigma`, `options.use_bias`, `options.activation`, and
`options.seed`/`options.seed2`; pass `options.remove_noise = true` for the
noise-free mean layer. `options.noise_kernel` and `options.noise_bias` inject
explicit noise tensors. The result table exposes `output`, the mean/sigma
variables, `variables`, and `initializers`. `tfa.layers.NoisyDense` is an
alias.

`stochastic_depth(graph, shortcut, residual, options)` follows TensorFlow
Addons' `[shortcut, residual]` contract. With `options.training = false` it
returns `shortcut + survival_probability * residual`; in training mode it keeps
the whole residual branch with probability `options.survival_probability`
(default `0.5`) and uses `options.seed` / `options.seed2` for the Bernoulli
draw. `options.training` may also be a boolean graph tensor.
`tfa.layers.StochasticDepth` is an alias.

### Optimizers

- `LAMB`
- `NovoGrad`
- `LazyAdam`
- `Yogi`
- `AdamW`
- `SGDW`
- `ConditionalGradient` (Frobenius norm)
- `MovingAverage` (wrapper)
- `Lookahead` (wrapper)
- `WeightDecay` (wrapper)

`LazyAdam(graph, loss, variables, options)` follows TensorFlow Addons'
sparse-friendly Adam semantics: moment accumulators and variables are only
updated where the gradient is non-zero, while the beta-power schedule advances
on every step. It accepts the same learning-rate, beta, epsilon, and
weight-decay options as `LAMB`.

`ConditionalGradient(graph, loss, variables, options)` applies the
Frobenius-norm conditional-gradient update
`learning_rate * variable - (1 - learning_rate) * lambda *
gradient / (||gradient|| + epsilon)`. It accepts `options.learning_rate`,
`options.lambda_` (or `options.lambda`), `options.epsilon`, and
`options.gradients`; `options.ord` currently supports only `"fro"`.

`Yogi(graph, loss, variables, options)` follows `tfa.optimizers.Yogi` with the
sign-based second-moment accumulator. It accepts `options.learning_rate`,
`options.beta1`, `options.beta2`, `options.epsilon`,
`options.initial_accumulator_value`, `options.activation = "sign" | "tanh"`,
and optional `options.l1_regularization_strength` /
`options.l2_regularization_strength` proximal terms.

`AdamW(graph, loss, variables, options)` exposes the core decoupled AdamW
update with TensorFlow Addons' default `weight_decay` of `0.004`.
`SGDW(graph, loss, variables, options)` exposes SGD with the same decoupled
weight-decay behavior; momentum and Nesterov options are not supported.

`MovingAverage(base_optimizer, options)` wraps a core optimizer and maintains a
shadow moving average of every variable with `options.average_decay` (default
`0.99`). `options.start_step`, `options.dynamic_decay`, and
`options.num_updates` follow TensorFlow Addons. The returned state includes
`average_variables` and `assign_averages`, a list of targets that copies the
averages back into the fast variables.

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
- `adjust_saturation`
- `adjust_hue`
- `adjust_hsv_in_yiq`
- `adjust_gamma`
- `adjust_contrast`
- `random_brightness`
- `random_contrast`
- `random_saturation`
- `random_hue`
- `random_crop`
- `rgb_to_hsv`
- `hsv_to_rgb`
- `rgb_to_grayscale`
- `grayscale_to_rgb`
- `transform`
- `compose_transforms`
- `angles_to_projective_transforms`
- `translations_to_projective_transforms`
- `flat_transforms_to_matrices`
- `matrices_to_flat_transforms`
- `shear_x`
- `shear_y`
- `cutout`
- `random_cutout`
- `resize_nearest`
- `bilinear_resize`
- `area_resize`
- `resize_bicubic`
- `crop_and_resize`
- `pad_to_bounding_box`
- `crop_to_bounding_box`
- `resize_with_pad`
- `central_crop`
- `dense_image_warp`
- `sharpness`
- `blend`
- `gaussian_filter2d`
- `median_filter2d`
- `mean_filter2d`
- `sobel_edges`
- `translate`
- `rotate`
- `random_hsv_in_yiq`

`adjust_saturation(graph, image, saturation, name)` scales HSV saturation,
following `tf.image.adjust_saturation`: `0` produces grayscale and `1`
preserves the image. `tfa.image.AdjustSaturation` is an alias.

`adjust_hue(graph, image, delta, name)` rotates hue by a normalized delta,
following `tf.image.adjust_hue`. `tfa.image.AdjustHue` is an alias.

`adjust_hsv_in_yiq(graph, image, delta_hue, scale_saturation, scale_value,
options)` follows `tfa.image.adjust_hsv_in_yiq`, applying hue rotation in radians
and saturation/value scaling in YIQ color space.
`tfa.image.AdjustHSVInYIQ` is an alias.

`adjust_gamma(graph, image, gamma, gain, name)` follows `tf.image.adjust_gamma`
with `gain * input^gamma`. `adjust_contrast(graph, image, contrast, name)`
follows `tf.image.adjust_contrast`, scaling each channel around its spatial
mean. `tfa.image.AdjustGamma` and `tfa.image.AdjustContrast` are aliases.

`random_brightness`, `random_contrast`, `random_saturation`, and `random_hue`
sample a single augmentation factor per call using `options.seed` /
`options.seed2`, matching TensorFlow's random color helpers. `random_crop`
returns a random contiguous `{height, width}` crop shared by the batch.

`rgb_to_hsv` and `hsv_to_rgb` convert rank-3+ RGB/HSV tensors.
`rgb_to_grayscale` uses the standard 0.2989/0.587/0.114 luminance weights and
`grayscale_to_rgb` replicates the grayscale channel. Their capitalized aliases
are provided for consistency.

`transform(graph, image, transforms, options)` follows `tfa.image.transform`.
Each transform row is `[a0, a1, a2, b0, b1, b2, c0, c1]` and maps output
coordinates back into the input. It accepts `options.output_shape`,
`options.interpolation`, `options.fill_mode`, and `options.fill_value`, and
supports rank-2 HW, rank-3 HWC, and rank-4 NHWC images.
`tfa.image.Transform` is an alias.

`compose_transforms(graph, transforms, options)` follows
`tfa.image.compose_transforms`, multiplying a non-empty list of flat eight-value
transform vectors or transform-row matrices into one composed transform.
`tfa.image.ComposeTransforms` is an alias.

`angles_to_projective_transforms(graph, angles, image_height, image_width,
options)` follows `tfa.image.angles_to_projective_transforms`, converting one
angle or a list of angles in radians into flat projective transform rows.
`tfa.image.AnglesToProjectiveTransforms` is an alias.

`translations_to_projective_transforms(graph, translations, options)` converts
one `{dx, dy}` pair or a list of pairs into flat transform rows.
`flat_transforms_to_matrices` appends the homogeneous coordinate, and
`matrices_to_flat_transforms` normalizes by the homogeneous entry. Their
capitalized aliases are available.

`shear_x(graph, image, level, options)` and `shear_y(graph, image, level,
options)` apply horizontal or vertical shear through the projective transform
op, honoring `options.interpolation`, `options.fill_mode`, and
`options.replace`. `tfa.image.ShearX` and `tfa.image.ShearY` are aliases.

`cutout(graph, images, mask_size, offset, options)` follows `tfa.image.cutout`
for rank-4 NHWC images. `mask_size` is a positive even `{height, width}` pair;
`offset` is a center pair shared by the batch or one pair per image.
`options.constant_values` controls the fill. `tfa.image.Cutout` is an alias.

`random_cutout(graph, images, mask_size, options)` follows
`tfa.image.random_cutout`, sampling a per-image center using
`options.seed`/`options.seed2`. `tfa.image.RandomCutout` is an alias.

`bilinear_resize(graph, image, height, width, name, options)` resizes rank-4
NHWC images with bilinear interpolation. By default it uses half-pixel centers,
matching `tf.image.resize(..., method = "bilinear")`; pass
`options.half_pixel_centers = false` or `options.align_corners = true` to
change the sampling convention. `area_resize(graph, image, height, width, name,
options)` uses area interpolation and averages the source pixels covered by
each destination pixel for float32/float64 images.
`tfa.image.BilinearResize` and `tfa.image.AreaResize` are aliases.

`sharpness(graph, image, factor, name, options)` follows `tfa.image.sharpness`
using PIL's 3x3 smoothing kernel. `factor = 1` preserves the image, lower
values blend toward the smoothed interior while retaining the original border,
and values above 1 extrapolate. It supports rank-3 HWC and rank-4 NHWC images
(`options.rank` for rank-3 inputs). `tfa.image.Sharpness` is an alias.

`dense_image_warp(graph, image, flow, options)` follows
`tfa.image.dense_image_warp`, sampling the source image at
`[row - flow_y, column - flow_x]` with bilinear interpolation and
nearest-border handling. `flow` has shape `[batch, height, width, 2]`.
`tfa.image.DenseImageWarp` is an alias.

`resize_bicubic(graph, image, height, width, name, options)` resizes rank-4
images with bicubic interpolation. `crop_and_resize(graph, image, boxes,
box_ind, crop_size, options)` samples normalized boxes with bilinear or nearest
interpolation. Their capitalized aliases are available.

`pad_to_bounding_box` and `crop_to_bounding_box` place or extract a static
rank-4 image rectangle. `resize_with_pad` preserves aspect ratio with bilinear
resize and centered padding, while `central_crop` extracts a centered fraction.
These helpers require `options.input_shape`. Their capitalized aliases are
available.

`blend(graph, image1, image2, factor, name)` follows `tfa.image.blend`:
factors between 0 and 1 interpolate and round, while factors above 1
extrapolate and clip to `0..255`. `tfa.image.Blend` is an alias.

`gaussian_filter2d(graph, image, options)` blurs rank-4 NHWC images (or rank-3
HWC images with `options.rank = 3`) using an odd-sized Gaussian kernel. It
accepts `filter_shape` (default `{ 3, 3 }`), `sigma` (number or
`{ sigma_y, sigma_x }`), and `padding` (`"REFLECT"` by default, plus
`"SYMMETRIC"`, `"CONSTANT"`, `"SAME"`, and `"VALID"`).

`median_filter2d(graph, image, options)` applies a median filter to rank-2,
rank-3, or rank-4 images (`options.rank`, default `4`). It accepts `filter_shape`
(default `{ 3, 3 }`, odd or even), `padding` (`"REFLECT"` by default, plus
`"SYMMETRIC"` and `"CONSTANT"`), and `constant_values` for constant padding.

`mean_filter2d(graph, image, options)` applies an average filter to rank-2,
rank-3, or rank-4 images (`options.rank`, default `4`). It accepts `filter_shape`
(default `{ 3, 3 }`, odd or even), `padding` (`"REFLECT"` by default, plus
`"SYMMETRIC"` and `"CONSTANT"`), and `constant_values` for constant padding.
Integer images are computed in float32 and cast back to their source type.

`sobel_edges(graph, image, options)` follows `tf.image.sobel_edges` for rank-4
NHWC `float32` or `float64` images. It reflect-pads the image by one pixel and
applies the vertical and horizontal Sobel filters per channel, returning a
tensor shaped `[batch, height, width, channels, 2]` whose final axis is
`[dy, dx]`.

`translate(graph, image, translations, options)` follows `tfa.image.translate`
for rank-2 HW, rank-3 HWC, or rank-4 NHWC images (pass `options.rank` for rank-3
or rank-2 inputs). A single `{ dx, dy }` pair applies to the whole input, or a
list of pairs supplies one translation per batch image. `options.interpolation`
is `"nearest"` (default) or `"bilinear"`; `options.fill_mode` is `"constant"`
(default), `"reflect"`, `"wrap"`, or `"nearest"`; and `options.fill_value` sets
the constant fill. `tfa.image.Translate` is an alias.

`rotate(graph, image, angles, options)` rotates rank-2 HW, rank-3 HWC, or
rank-4 NHWC images counterclockwise by radians (`options.rank` is required for
rank-3/rank-2 inputs). A scalar angle applies to the whole input, or a list
supplies one angle per batch image. It accepts the same `interpolation`,
`fill_mode`, and `fill_value` options as `translate`.
`tfa.image.Rotate` is an alias.

`random_hsv_in_yiq(graph, image, options)` follows
`tfa.image.random_hsv_in_yiq` for rank-3 or rank-4 float RGB images. It samples
a hue delta in `[-max_delta_hue, max_delta_hue]` radians and saturation/value
scales from `lower_saturation`/`upper_saturation` and
`lower_value`/`upper_value` (all default to `1`, with `max_delta_hue` default
`0`), using `options.seed`/`options.seed2` for reproducibility.
`tfa.image.RandomHSVInYIQ` is an alias.

### Callbacks

- `TimeStopping`
- `TerminateOnNaN`
- `BackupAndRestore`
- `CSVLogger`
- `TQDMProgressBar` (plain Lua progress output)

`TQDMProgressBar(options)` provides a dependency-free progress callback with
train/epoch lifecycle hooks, metric formatting, and optional progress output.
It never imports Python or an external tqdm package.

## Parity status

| Area | Implemented | Remaining TensorFlow Addons work |
| --- | --- | --- |
| Activations | `gelu`, `swish`, `mish`, `lisht`, `tanhshrink`, `snake` | additional activation wrappers |
| Losses | `focal`, `dice`, `tversky`, `triplet`, `contrastive`, `giou`, `log_cosh`, `hinge`, `kl_divergence`, `pinball`, `sigmoid_focal_crossentropy`, `npairs_loss`, `npairs_multilabel_loss`, `triplet_semihard_loss`, `lifted`, `weighted_kappa`, `sparsemax` | hard triplet and further specialized losses |
| Metrics | `f1_score`, `precision`, `recall`, `fbeta_score`, `matthews_corrcoef`, `cohen_kappa`, `r_square`, `pearson_correlation`, `mean_iou`, `multilabel_confusion_matrix`, `hamming_loss`, `hamming_distance`, `auc`, `sensitivity_at_specificity`, `specificity_at_sensitivity`, `precision_at_recall`, `recall_at_precision`, `geometric_mean`, `harmonic_mean`, `multiclass_f1` | further classification metrics |
| Normalization layers | `group`, `instance`, `layer`, `weight`, `spectral`, `filter_response_normalization`, `poincare_normalize`, `rms_normalization`, `unit_normalization`, `standardize`, `local_response_normalization` | further normalization variants |
| Other layers | `poisson`, `quantization`, `maxout`, `tlu`, `polynomial_crossing`, `sparsemax`, `adaptive_average_pooling_1d`, `adaptive_max_pooling_1d`, `adaptive_average_pooling_2d`, `adaptive_max_pooling_2d`, `adaptive_average_pooling_3d`, `adaptive_max_pooling_3d`, `netvlad`, `max_unpooling_2d`, `snake`, `gelu`, `spatial_pyramid_pooling_2d`, `esn`, `multi_head_attention`, `noisy_dense`, `stochastic_depth` | further layers |
| Optimizers | `LAMB`, `NovoGrad`, `LazyAdam`, `Yogi`, `AdamW`, `SGDW`, `ConditionalGradient` (Frobenius norm), `MovingAverage`, `Lookahead`, `WeightDecay` | nuclear-norm conditional gradient, SGDW momentum, and further wrappers |
| Image | horizontal/vertical flips, brightness, contrast, saturation/hue, random color/crop augmentation, HSV-in-YIQ, gamma, RGB/HSV/grayscale conversion, nearest/bilinear/bicubic/area resize, `crop_and_resize`, bounding-box pad/crop/resize, `central_crop`, `sharpness`, `blend`, `dense_image_warp`, `gaussian_filter2d`, `median_filter2d`, `mean_filter2d`, `sobel_edges`, `transform`, `compose_transforms`, `angles_to_projective_transforms`, `translations_to_projective_transforms`, `flat_transforms_to_matrices`, `matrices_to_flat_transforms`, `shear_x`, `shear_y`, `translate`, `rotate`, `cutout`, `random_cutout`, `random_hsv_in_yiq` | further warps and color transforms |
| Callbacks | `TimeStopping`, `TerminateOnNaN`, `BackupAndRestore`, `CSVLogger`, `TQDMProgressBar` | further progress reporting |
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
