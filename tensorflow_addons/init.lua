--
-- TensorFlow Addons for LuaJIT.
--
local Addons = {
  VERSION = "0.12.0",
  activations = require("tensorflow_addons.activations"),
  losses = require("tensorflow_addons.losses"),
  metrics = require("tensorflow_addons.metrics"),
  layers = require("tensorflow_addons.layers"),
  callbacks = require("tensorflow_addons.callbacks"),
  optimizers = require("tensorflow_addons.optimizers"),
  image = require("tensorflow_addons.image"),
}

return Addons
