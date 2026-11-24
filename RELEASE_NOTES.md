# tensorflow-lua-addons 0.13.0

- binary hinge loss with configurable margin and `none`, `mean`, and `sum`
  reductions;
- KL divergence loss over last-axis probability distributions, with finite
  handling for zero labels and predictions;
- precision and recall metrics with binary thresholding and macro, micro, or
  weighted multiclass averages;
- `sobel_edges` image helper with reflected padding and per-channel `[dy, dx]`
  edge maps;
- 60 graph-mode tests.
