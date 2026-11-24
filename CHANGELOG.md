# Changelog

## 0.14.0

- binary ROC/PR AUC with configurable thresholds, summation methods, and
  logits;
- sensitivity-at-specificity and specificity-at-sensitivity operating-point
  metrics;
- NoisyDense layer with factorised or independent Gaussian noise and optional
  noise removal;
- `translate` image transform with nearest/bilinear interpolation, fill modes,
  constant fill values, and per-image translations;
- 66 graph-mode tests.

## 0.13.0

- binary hinge loss with configurable margin and `none`, `mean`, and `sum`
  reductions;
- KL divergence loss over last-axis probability distributions, with finite
  handling for zero labels and predictions;
- precision and recall metrics with binary thresholding and macro, micro, or
  weighted multiclass averages;
- `sobel_edges` image helper with reflected padding and per-channel `[dy, dx]`
  edge maps;
- 60 graph-mode tests.

## 0.12.0

- N-pairs loss and multilabel N-pairs loss with stable softmax cross entropy;
- semi-hard triplet loss with `L2`, `squared-L2`, and `angular` distance
  metrics;
- filter response normalization layer with optional learned epsilon;
- Hamming loss and Hamming distance metrics for multiclass and multilabel
  inputs;
- `mean_filter2d` for rank-2, rank-3, and rank-4 images, including even filter
  sizes;
- 55 graph-mode tests.

## 0.11.0

- pinball/quantile loss with `none`, `mean`, and `sum` reductions;
- sigmoid focal cross entropy with `alpha`, `gamma`, `from_logits`, and
  reduction options;
- multilabel confusion matrix metric with per-class `[[TN, FP], [FN, TP]]`
  matrices;
- `median_filter2d` for rank-2, rank-3, and rank-4 images, including even filter
  sizes;
- stochastic depth layer with training/inference and seeded Bernoulli behavior;
- 49 graph-mode tests.

## 0.10.0

- LazyAdam optimizer with lazy moment and weight updates;
- stable `log_cosh` loss with `none`, `mean`, and `sum` reductions;
- `mean_iou` metric for integer and one-hot class inputs;
- `gaussian_filter2d` image helper;
- Maxout layer;
- multiclass `f1_score` with macro, micro, and weighted averages;
- multiclass `cohen_kappa` with linear and quadratic weights;
- register Poisson and quantization layers at module load time;
- 42 tests.

## 0.9.0

- WeightDecay optimizer wrapper;
- Poisson layer;
- fake-quantization layer;
- 26 tests.

## 0.8.0

- image helpers.
