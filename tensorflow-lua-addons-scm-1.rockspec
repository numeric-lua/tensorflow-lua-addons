package = "tensorflow-lua-addons"
version = "scm-1"

source = {
  url = "git+https://github.com/numeric-lua/tensorflow-lua-addons.git",
}

description = {
  summary = "TensorFlow Addons for LuaJIT",
  detailed = [[
    Activations including Snake, hinge/KL/contrastive/N-pairs losses, AUC and
    operating-point metrics, filter-response, Poincare, TLU, PolynomialCrossing,
    and NoisyDense layers, stochastic depth, LAMB, NovoGrad, LazyAdam, Yogi,
    ConditionalGradient, MovingAverage, Lookahead, image filters, sharpness,
    resizing, translation, rotation, and color augmentation, and callbacks built
    on tensorflow-lua. Requires a local libtensorflow installation.
  ]],
  homepage = "https://github.com/numeric-lua/tensorflow-lua-addons",
  license = "BSD-3-Clause",
}

dependencies = {
  "lua >= 5.1",
  "tensorflow-lua >= 0.8",
}

build = {
  type = "builtin",
  modules = {
    tensorflow_addons = "tensorflow_addons/init.lua",
    ["tensorflow_addons.activations"] = "tensorflow_addons/activations.lua",
    ["tensorflow_addons.losses"] = "tensorflow_addons/losses.lua",
    ["tensorflow_addons.metrics"] = "tensorflow_addons/metrics.lua",
    ["tensorflow_addons.layers"] = "tensorflow_addons/layers.lua",
    ["tensorflow_addons.callbacks"] = "tensorflow_addons/callbacks.lua",
    ["tensorflow_addons.optimizers"] = "tensorflow_addons/optimizers.lua",
    ["tensorflow_addons.image"] = "tensorflow_addons/image.lua",
    ["tensorflow_addons.ops"] = "tensorflow_addons/ops.lua",
  },
}
