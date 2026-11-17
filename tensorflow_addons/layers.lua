local tf = require("tensorflow")
local Ops = require("tensorflow_addons.ops")

local Layers = {}

local function ones(count)
  local values = {}
  for index = 1, count do values[index] = 1 end
  return values
end

local function zeros(count)
  local values = {}
  for index = 1, count do values[index] = 0 end
  return values
end

local function normalize(graph, input, axes, name, dtype, options)
  local epsilon = options.epsilon or 1e-5
  local mean = tf.reduce_mean(graph, input, axes, name .. "/mean", dtype, true)
  local centered = tf.sub(graph, input, mean, name .. "/centered", dtype)
  local variance = tf.reduce_mean(graph, tf.square(graph, centered,
    name .. "/squared", dtype), axes, name .. "/variance", dtype, true)
  local epsilon_op = tf.constant(graph, epsilon, name .. "/epsilon", dtype)
  local safe_variance = Ops.maximum(graph, variance, epsilon_op,
    name .. "/safe_variance", dtype)
  local denominator = tf.sqrt(graph, tf.add(graph, safe_variance, epsilon_op,
    name .. "/denominator_base", dtype), name .. "/denominator", dtype)
  return tf.div(graph, centered, denominator, name .. "/normalized", dtype)
end

local function affine(graph, normalized, channels, name, dtype)
  local gamma = tf.variable(graph, ones(channels), name .. "/gamma",
    { dtype = dtype })
  local beta = tf.variable(graph, zeros(channels), name .. "/beta",
    { dtype = dtype })
  local scaled = tf.mul(graph, normalized, gamma.read, name .. "/scaled",
    dtype)
  local output = tf.add(graph, scaled, beta.read, name .. "/output", dtype)
  return output, gamma, beta
end

function Layers.group_normalization(graph, input, groups, options)
  options = options or {}
  local shape = options.input_shape
  if type(shape) ~= "table" then
    error("group_normalization requires options.input_shape", 3)
  end
  local dtype = options.dtype or "float32"
  local prefix = options.name or "group_normalization"
  local channels = shape[#shape]
  if channels % groups ~= 0 then error("groups must divide channels", 3) end
  local grouped, axes, restore
  if #shape == 1 then
    grouped = tf.reshape(graph, input, { -1, groups, channels / groups },
      prefix .. "/grouped", dtype)
    axes = { 0, 2 }
    restore = { -1, channels }
  elseif #shape == 3 then
    grouped = tf.reshape(graph, input,
      { -1, shape[1], shape[2], groups, channels / groups },
      prefix .. "/grouped", dtype)
    axes = { 1, 2, 4 }
    restore = { -1, shape[1], shape[2], channels }
  else
    error("group_normalization supports rank-2 or NHWC rank-4 inputs", 3)
  end
  local normalized = normalize(graph, grouped, axes, prefix, dtype, options)
  normalized = tf.reshape(graph, normalized, restore, prefix .. "/ungrouped",
    dtype)
  local output, gamma, beta = affine(graph, normalized, channels, prefix, dtype)
  return {
    output = output, gamma = gamma, beta = beta, variables = { gamma, beta },
    initializers = { gamma.initializer, beta.initializer }, channels = channels,
  }
end

function Layers.instance_normalization(graph, input, options)
  options = options or {}
  local shape = options.input_shape
  if type(shape) ~= "table" then
    error("instance_normalization requires options.input_shape", 3)
  end
  local channels = shape[#shape]
  return Layers.group_normalization(graph, input, channels,
    { input_shape = shape, epsilon = options.epsilon,
      dtype = options.dtype, name = options.name or "instance_normalization" })
end

function Layers.layer_normalization(graph, input, options)
  options = options or {}
  local shape = options.input_shape
  if type(shape) ~= "table" then
    error("layer_normalization requires options.input_shape", 3)
  end
  local dtype = options.dtype or "float32"
  local prefix = options.name or "layer_normalization"
  local channels = shape[#shape]
  local normalized = normalize(graph, input, { #shape - 1 }, prefix, dtype,
    options)
  local output, gamma, beta = affine(graph, normalized, channels, prefix, dtype)
  return {
    output = output, gamma = gamma, beta = beta, variables = { gamma, beta },
    initializers = { gamma.initializer, beta.initializer }, channels = channels,
  }
end

function Layers.weight_normalization(graph, shape, options)
  options = options or {}
  if type(shape) ~= "table" or #shape < 1 then
    error("weight_normalization requires a weight shape", 3)
  end
  local dtype = options.dtype or "float32"
  local prefix = options.name or "weight_normalization"
  local bound = math.sqrt(6 / (shape[1] * (shape[#shape] or 1)))
  local function random_tensor(dimension)
    if dimension > #shape then return (math.random() * 2 - 1) * bound end
    local result = {}
    for index = 1, shape[dimension] do
      result[index] = random_tensor(dimension + 1)
    end
    return result
  end
  local v = tf.variable(graph, random_tensor(1), prefix .. "/v",
    { dtype = dtype })
  local g_values = {}
  for index = 1, shape[#shape] do g_values[index] = 1 end
  local g = tf.variable(graph, g_values, prefix .. "/g", { dtype = dtype })
  local flat = tf.reshape(graph, v.read, { -1, shape[#shape] },
    prefix .. "/flat", dtype)
  local squared = tf.square(graph, flat, prefix .. "/squared", dtype)
  local variance = tf.reduce_sum(graph, squared, { 0 }, prefix .. "/norm",
    dtype, true)
  local denominator = tf.sqrt(graph, tf.add(graph, variance,
    tf.constant(graph, options.epsilon or 1e-12, prefix .. "/epsilon", dtype),
    prefix .. "/denominator_base", dtype), prefix .. "/denominator", dtype)
  local normalized = tf.div(graph, flat, denominator, prefix .. "/normalized",
    dtype)
  local scaled = tf.mul(graph, normalized, g.read, prefix .. "/scaled", dtype)
  local output = tf.reshape(graph, scaled, shape, prefix .. "/weight", dtype)
  return {
    output = output, v = v, g = g, variables = { v, g },
    initializers = { v.initializer, g.initializer }, shape = shape,
  }
end

return Layers
