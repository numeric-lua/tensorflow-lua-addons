# Changelog

## Unreleased

- Sparsemax layer for sparse simplex projections along a selected axis;
- generic projective `transform` image helper with configurable output shape;
- `cutout` image mask with explicit per-image centers and fill values;
- `random_cutout` image mask with seeded per-image centers;
- `adjust_hue` image color helper;
- adaptive average and max 2D pooling layers;
- AdamW and SGDW decoupled weight-decay optimizers;
- `blend` image helper with interpolating and extrapolating factors;
- geometric and harmonic mean metrics;
- `class_id` selection for precision, recall, and F1 score;
- adaptive average and max 1D pooling layers;
- `adjust_hsv_in_yiq` color adjustment helper;
- `compose_transforms` helper for projective transform composition;
- `angles_to_projective_transforms` helper for rotation matrices;
- lifted structured loss;
- weighted kappa loss;
- sparsemax loss from logits;
- `adjust_gamma` and `adjust_contrast` image helpers;
- adaptive average and max 3D pooling layers;
- NetVLAD aggregation layer;
- MaxUnpooling2D layer;
- trainable-frequency Snake layer wrapper;
- RGB/HSV and RGB/grayscale conversion helpers;
- `dense_image_warp` bilinear flow warping helper;
- translation-to-projective-transform helper;
- flat-transform/matrix conversion helpers;
- horizontal and vertical image shear helpers;
- random brightness, contrast, saturation, and hue augmentation helpers;
- random image crop helper;
- GELU and SpatialPyramidPooling2D layers;
- dependency-free TQDMProgressBar callback;
- bicubic resize and normalized crop-and-resize image helpers;
- ESN recurrent layer with fixed reservoir;
- bounding-box pad/crop and centered resize/crop image helpers;
- 135 graph-mode tests.

## 0.17.0

- Snake periodic activation with configurable frequency;
- TLU layer with learned thresholds and optional affine slopes;
- PolynomialCrossing layer with full-rank or low-rank Deep & Cross projection;
- `sharpness` image helper using the PIL smoothing kernel and blend factor;
- Yogi adaptive optimizer with sign/tanh second-moment updates and optional
  proximal regularization;
- 80 graph-mode tests.

## 0.16.0

- Poincare normalization layer for projecting slices into the unit ball;
- bilinear image resize with configurable pixel-center and corner alignment;
- area image resize with source-pixel averaging;
- HSV saturation adjustment helper for RGB images;
- MovingAverage optimizer wrapper with averaged-weight assignment targets;
- 75 graph-mode tests.

## 0.15.0

- precision-at-recall operating-point metric;
- recall-at-precision operating-point metric;
- ConditionalGradient optimizer with the Frobenius-norm constraint;
- `rotate` image transform with per-image angles and configurable fill modes;
- random HSV color augmentation in YIQ space with seeded ranges;
- 70 graph-mode tests.

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
