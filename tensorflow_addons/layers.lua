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

local function filled(shape, value, depth)
  depth = depth or 1
  if depth > #shape then return value end
  local result = {}
  for index = 1, shape[depth] do
    result[index] = filled(shape, value, depth + 1)
  end
  return result
end

local function pack_int32(graph, values, name)
  local inputs = {}
  for index, value in ipairs(values) do
    if type(value) == "number" then
      inputs[index] = tf.constant(graph, value, name .. "/" .. index, "int32")
    else
      inputs[index] = value
    end
  end
  local desc = graph:new_operation("Pack", name)
  desc:add_inputs(inputs)
  desc:set_type("T", "int32")
  desc:set_int("axis", 0)
  desc:set_int("N", #inputs)
  return desc:finish()
end

local function last_dimension(graph, tensor, name, dtype)
  local shape = tf.shape(graph, tensor, name .. "/shape", dtype)
  local begin = tf.constant(graph, { -1 }, name .. "/begin", "int32")
  local limit = tf.constant(graph, { 2147483647 }, name .. "/end", "int32")
  local stride = tf.constant(graph, { 1 }, name .. "/stride", "int32")
  local desc = graph:new_operation("StridedSlice", name .. "/slice")
  desc:add_input(shape)
  desc:add_input(begin)
  desc:add_input(limit)
  desc:add_input(stride)
  desc:set_type("T", "int32")
  desc:set_type("Index", "int32")
  desc:set_int("begin_mask", 0)
  desc:set_int("end_mask", 0)
  desc:set_int("ellipsis_mask", 0)
  desc:set_int("new_axis_mask", 0)
  desc:set_int("shrink_axis_mask", 0)
  return tf.reshape(graph, desc:finish(), {}, name .. "/scalar", "int32")
end

local function shape_prefix(graph, shape, name)
  local begin = tf.constant(graph, { 0 }, name .. "/begin", "int32")
  local limit = tf.constant(graph, { -1 }, name .. "/end", "int32")
  local stride = tf.constant(graph, { 1 }, name .. "/stride", "int32")
  local desc = graph:new_operation("StridedSlice", name .. "/slice")
  desc:add_input(shape)
  desc:add_input(begin)
  desc:add_input(limit)
  desc:add_input(stride)
  desc:set_type("T", "int32")
  desc:set_type("Index", "int32")
  desc:set_int("begin_mask", 0)
  desc:set_int("end_mask", 0)
  desc:set_int("ellipsis_mask", 0)
  desc:set_int("new_axis_mask", 0)
  desc:set_int("shrink_axis_mask", 0)
  return desc:finish()
end

local function floor_div_int32(graph, a, b, name)
  local desc = graph:new_operation("FloorDiv", name)
  desc:add_input(a)
  desc:add_input(b)
  desc:set_type("T", "int32")
  return desc:finish()
end

local function reshape_dynamic(graph, input, shape, name, dtype)
  local desc = graph:new_operation("Reshape", name)
  desc:add_input(input)
  desc:add_input(shape)
  desc:set_type("T", dtype)
  desc:set_type("Tshape", "int32")
  return desc:finish()
end

local function maxout_last(graph, input, num_units, name, dtype)
  local shape = tf.shape(graph, input, name .. "/shape", dtype)
  local channels = last_dimension(graph, input, name .. "/channels", dtype)
  local units = tf.constant(graph, num_units, name .. "/units", "int32")
  local per_group = floor_div_int32(graph, channels, units,
    name .. "/per_group")
  local tail = pack_int32(graph, { units, per_group },
    name .. "/tail_shape")
  local grouped_shape = tf.concat(graph, {
    shape_prefix(graph, shape, name .. "/prefix"),
    tail,
  }, 0, name .. "/grouped_shape", "int32")
  local grouped = reshape_dynamic(graph, input, grouped_shape,
    name .. "/grouped", dtype)
  return tf.reduce_max(graph, grouped, { -1 }, name .. "/output", dtype, false)
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

-- Poisson layer.  Sampling is deterministic unless the caller supplies
-- options.noise; with noise it applies the standard inverse-CDF transform.
function Layers.poisson(graph, log_rate, options)
  options = options or {}
  local name = options.name or "poisson"
  local dtype = options.dtype or "float32"
  local rate = tf.exp(graph, log_rate, name .. "/rate", dtype)
  if options.noise == nil then return rate end
  local transformed = tf.sub(graph, tf.constant(graph, 1, name .. "/one", dtype),
    options.noise, name .. "/complement", dtype)
  return tf.mul(graph, tf.constant(graph, -1, name .. "/negative", dtype),
    tf.log(graph, tf.maximum(graph, transformed,
      tf.constant(graph, 1e-7, name .. "/epsilon", dtype), name .. "/safe",
      dtype), name .. "/log", dtype), name, dtype)
end

function Layers.quantization(graph, input, options)
  options = options or {}
  local name = options.name or "quantization"
  local dtype = options.dtype or "float32"
  local min_value = options.min_value or -6
  local max_value = options.max_value or 6
  local desc = graph:new_operation("FakeQuantWithMinMaxVars", name)
  desc:add_input(input)
  desc:add_input(tf.constant(graph, min_value, name .. "/min", dtype))
  desc:add_input(tf.constant(graph, max_value, name .. "/max", dtype))
  desc:set_int("num_bits", options.num_bits or 8)
  desc:set_bool("narrow_range", options.narrow_range == true)
  return desc:finish()
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


local function matmul(graph, a, b, transpose_a, transpose_b, name, dtype)
local desc = graph:new_operation("MatMul", name)
desc:add_input(a)
desc:add_input(b)
desc:set_type("T", dtype)
desc:set_bool("transpose_a", transpose_a == true)
desc:set_bool("transpose_b", transpose_b == true)
return desc:finish()
end

local function assign(graph, ref, value, name, dtype)
local desc = graph:new_operation("Assign", name)
desc:add_input(ref)
desc:add_input(value)
desc:set_type("T", dtype)
desc:set_bool("use_locking", true)
return desc:finish()
end

local function l2_normalize(graph, value, name, dtype, epsilon)
local squared = tf.square(graph, value, name .. "/squared", dtype)
local total = tf.reduce_sum(graph, squared, { 0, 1 }, name .. "/sum", dtype)
local norm = tf.sqrt(graph, tf.add(graph, total,
  tf.constant(graph, epsilon or 1e-12, name .. "/epsilon", dtype),
  name .. "/norm_base", dtype), name .. "/norm", dtype)
return tf.div(graph, value, norm, name .. "/normalized", dtype)
end

function Layers.spectral_normalization(graph, weight, options)
options = options or {}
local shape = options.weight_shape
if type(shape) ~= "table" or #shape ~= 2 then
  error("spectral_normalization requires options.weight_shape = {m, n}", 3)
end
local m, n = shape[1], shape[2]
local dtype = options.dtype or "float32"
local prefix = options.name or "spectral_normalization"
local iterations = options.iterations or 1
if type(iterations) ~= "number" or iterations < 0 then
  error("spectral_normalization iterations must be non-negative", 3)
end
local function initial_vector(count)
  local value = 1 / math.sqrt(count)
  local result = {}
  for index = 1, count do result[index] = { value } end
  return result
end
local u = tf.variable(graph, initial_vector(m), prefix .. "/u",
  { dtype = dtype })
local v = tf.variable(graph, initial_vector(n), prefix .. "/v",
  { dtype = dtype })
local u_value, v_value = u.read, v.read
for iteration = 1, iterations do
  local wt_u = matmul(graph, weight, u_value, true, false,
    prefix .. "/wt_u_" .. iteration, dtype)
  v_value = l2_normalize(graph, wt_u, prefix .. "/v_" .. iteration, dtype,
    options.epsilon)
  local w_v = matmul(graph, weight, v_value, false, false,
    prefix .. "/w_v_" .. iteration, dtype)
  u_value = l2_normalize(graph, w_v, prefix .. "/u_" .. iteration, dtype,
    options.epsilon)
end
local wv = matmul(graph, weight, v_value, false, false,
  prefix .. "/final_wv", dtype)
local sigma = tf.reduce_sum(graph, tf.mul(graph, u_value, wv,
  prefix .. "/sigma_tensor", dtype), { 0, 1 }, prefix .. "/sigma", dtype)
local safe_sigma = Ops.maximum(graph, sigma,
  tf.constant(graph, options.epsilon or 1e-12, prefix .. "/sigma_epsilon",
    dtype), prefix .. "/safe_sigma", dtype)
local output = tf.div(graph, weight, safe_sigma, prefix .. "/weight", dtype)
local updates = {
  assign(graph, u.ref, u_value, prefix .. "/update_u", dtype),
  assign(graph, v.ref, v_value, prefix .. "/update_v", dtype),
}
return {
  output = output, u = u, v = v, variables = { u, v },
  initializers = { u.initializer, v.initializer }, updates = updates,
  sigma = sigma, weight_shape = shape,
}
end

-- Maxout layer.  The selected axis is moved to the end, its channels are
-- grouped into num_units groups, and the maximum of each group is returned.
function Layers.maxout(graph, input, num_units, options)
  options = options or {}
  local name = options.name or "maxout"
  local dtype = options.dtype or input.dtype or "float32"
  if type(num_units) ~= "number" or num_units % 1 ~= 0 or num_units < 1 then
    error("maxout requires num_units to be a positive integer", 3)
  end
  local axis = options.axis or -1
  local shape = options.input_shape
  local rank = type(shape) == "table" and #shape or nil
  local inverse
  if axis ~= -1 and rank == nil then
    error("maxout requires options.input_shape when axis is not -1", 3)
  end
  if rank ~= nil then
    if type(axis) ~= "number" or axis % 1 ~= 0 then
      error("maxout axis must be an integer", 3)
    end
    if axis < 0 then axis = axis + rank end
    if axis < 0 or axis >= rank then
      error("maxout axis is out of range", 3)
    end
    local channels = shape[axis + 1]
    if channels % num_units ~= 0 then
      error("maxout num_units must divide the selected axis size", 3)
    end
    if axis ~= rank - 1 then
      local perm = {}
      inverse = {}
      for index = 0, rank - 1 do
        if index ~= axis then perm[#perm + 1] = index end
      end
      perm[#perm + 1] = axis
      for index = 1, rank do inverse[perm[index] + 1] = index - 1 end
      input = tf.transpose(graph, input, perm, name .. "/transpose", dtype)
    end
  end
  local output = maxout_last(graph, input, num_units, name, dtype)
  if inverse ~= nil then
    output = tf.transpose(graph, output, inverse, name .. "/untranspose", dtype)
  end
  return output
end

Layers.Maxout = Layers.maxout

local function random_uniform_scalar(graph, name, dtype, seed, seed2)
  local shape = tf.constant(graph, {}, name .. "/shape", "int32")
  local desc = graph:new_operation("RandomUniform", name)
  desc:add_input(shape)
  desc:set_type("dtype", dtype)
  desc:set_int("seed", seed or 0)
  desc:set_int("seed2", seed2 or 0)
  return desc:finish()
end

local function less_scalar(graph, value, threshold, name, dtype)
  local desc = graph:new_operation("Less", name)
  desc:add_input(value)
  desc:add_input(threshold)
  desc:set_type("T", dtype)
  return desc:finish()
end

-- Stochastic depth, matching tfa.layers.StochasticDepth's [shortcut, residual]
-- semantics.  Training keeps the whole residual branch with probability
-- survival_probability; inference scales the residual by that probability.
function Layers.stochastic_depth(graph, shortcut, residual, options)
  options = options or {}
  local name = options.name or "stochastic_depth"
  local dtype = options.dtype or shortcut.dtype or "float32"
  local probability = options.survival_probability
  if probability == nil then probability = 0.5 end
  if type(probability) ~= "number" or probability ~= probability or
      probability < 0 or probability > 1 then
    error("stochastic_depth survival_probability must be in [0, 1]", 3)
  end
  local shortcut_value = tf.cast(graph, shortcut,
    shortcut.dtype or dtype, dtype, name .. "/shortcut")
  local residual_value = tf.cast(graph, residual,
    residual.dtype or dtype, dtype, name .. "/residual")
  local probability_op = tf.constant(graph, probability,
    name .. "/survival_probability", dtype)
  local scaled_residual = tf.mul(graph, probability_op, residual_value,
    name .. "/scaled_residual", dtype)
  local inference = tf.add(graph, shortcut_value, scaled_residual,
    name .. "/inference", dtype)
  local training = options.training
  if training == false then return inference end
  local random = random_uniform_scalar(graph, name .. "/random", dtype,
    options.seed, options.seed2)
  local mask = less_scalar(graph, random, probability_op, name .. "/mask",
    dtype)
  local mask_value = tf.cast(graph, mask, "bool", dtype, name .. "/mask_value")
  local residual_branch = tf.mul(graph, mask_value, residual_value,
    name .. "/residual_branch", dtype)
  local train_output = tf.add(graph, shortcut_value, residual_branch,
    name .. "/training", dtype)
  if type(training) == "table" and training.oper ~= nil then
    return tf.where(graph, training, train_output, inference, name, dtype)
  end
  return train_output
end

Layers.StochasticDepth = Layers.stochastic_depth

-- Filter response normalization, matching tfa.layers.FilterResponseNormalization.
-- The layer normalizes each channel over the spatial axes [1, 2] of an NHWC
-- tensor and applies per-channel scale and offset.  options.learned_epsilon
-- adds the trainable positive epsilon term from TensorFlow Addons.
function Layers.filter_response_normalization(graph, input, options)
  options = options or {}
  local shape = options.input_shape
  if type(shape) ~= "table" or #shape ~= 4 then
    error("filter_response_normalization requires a rank-4 input_shape", 3)
  end
  local dtype = options.dtype or input.dtype or "float32"
  local prefix = options.name or "filter_response_normalization"
  local axis = options.axis or { 1, 2 }
  if type(axis) ~= "table" or #axis ~= 2 then
    error("filter_response_normalization axis must be { 1, 2 }", 3)
  end
  local resolved = {}
  for index = 1, #axis do
    local value = axis[index]
    if type(value) ~= "number" or value % 1 ~= 0 then
      error("filter_response_normalization axis entries must be integers", 3)
    end
    if value < 0 then value = value + #shape end
    resolved[index] = value
  end
  if resolved[1] ~= 1 or resolved[2] ~= 2 then
    error("filter_response_normalization operates on spatial axes { 1, 2 }", 3)
  end
  local epsilon = options.epsilon
  if epsilon == nil then epsilon = 1e-6 end
  if type(epsilon) ~= "number" or epsilon ~= epsilon or
      epsilon == math.huge or epsilon == -math.huge then
    error("filter_response_normalization epsilon must be a finite number", 3)
  end
  local channels = shape[4]
  local gamma = tf.variable(graph, filled({ 1, 1, 1, channels }, 1),
    prefix .. "/gamma", { dtype = dtype })
  local beta = tf.variable(graph, filled({ 1, 1, 1, channels }, 0),
    prefix .. "/beta", { dtype = dtype })
  local variables = { gamma, beta }
  local initializers = { gamma.initializer, beta.initializer }
  local squared = tf.square(graph, input, prefix .. "/squared", dtype)
  local nu2 = tf.reduce_mean(graph, squared, { 1, 2 }, prefix .. "/nu2", dtype,
    true)
  local epsilon_op = tf.constant(graph, math.abs(epsilon),
    prefix .. "/epsilon", dtype)
  local epsilon_total = epsilon_op
  local learned_epsilon
  if options.learned_epsilon == true then
    learned_epsilon = tf.variable(graph, { 1e-4 },
      prefix .. "/learned_epsilon", { dtype = dtype })
    local absolute = tf.abs(graph, learned_epsilon.read,
      prefix .. "/learned_epsilon_abs", dtype)
    epsilon_total = tf.add(graph, epsilon_op, absolute,
      prefix .. "/epsilon_total", dtype)
    variables[#variables + 1] = learned_epsilon
    initializers[#initializers + 1] = learned_epsilon.initializer
  end
  local denominator_base = tf.add(graph, nu2, epsilon_total,
    prefix .. "/denominator_base", dtype)
  local denominator = tf.sqrt(graph, denominator_base,
    prefix .. "/denominator", dtype)
  local normalized = tf.div(graph, input, denominator,
    prefix .. "/normalized", dtype)
  local scaled = tf.mul(graph, gamma.read, normalized, prefix .. "/scaled",
    dtype)
  local output = tf.add(graph, scaled, beta.read, prefix .. "/output", dtype)
  return {
    output = output,
    gamma = gamma,
    beta = beta,
    learned_epsilon = learned_epsilon,
    variables = variables,
    initializers = initializers,
    axis = resolved,
    epsilon = epsilon,
  }
end

Layers.FilterResponseNormalization = Layers.filter_response_normalization

local function random_uniform_value(bound)
  return (math.random() * 2 - 1) * bound
end

local function random_array(shape, bound, depth)
  depth = depth or 1
  if depth > #shape then return random_uniform_value(bound) end
  local result = {}
  for index = 1, shape[depth] do
    result[index] = random_array(shape, bound, depth + 1)
  end
  return result
end

local function random_standard_normal(graph, shape, name, dtype, seed, seed2)
  local desc = graph:new_operation("RandomStandardNormal", name)
  desc:add_input(tf.constant(graph, shape, name .. "/shape", "int32"))
  desc:set_type("dtype", dtype)
  desc:set_int("seed", seed or 0)
  desc:set_int("seed2", seed2 or 0)
  return desc:finish()
end

local function scaled_noise(graph, shape, name, dtype, seed, seed2)
  local noise = random_standard_normal(graph, shape, name .. "/normal", dtype,
    seed, seed2)
  local absolute = tf.abs(graph, noise, name .. "/absolute", dtype)
  local root = tf.sqrt(graph, absolute, name .. "/root", dtype)
  local zero = tf.constant(graph, 0, name .. "/zero", dtype)
  local nonnegative = tf.greater_equal(graph, noise, zero,
    name .. "/nonnegative", dtype)
  local negative = tf.mul(graph, tf.constant(graph, -1,
    name .. "/negative_one", dtype), root, name .. "/negative", dtype)
  return tf.where(graph, nonnegative, root, negative, name, dtype)
end

local function noisy_dense_activation(graph, input, activation, name, dtype)
  if activation == nil or activation == "linear" then return input end
  if type(activation) == "function" then
    return activation(graph, input, name, dtype)
  end
  local supported = {
    relu = tf.nn.relu,
    relu6 = tf.nn.relu6,
    sigmoid = tf.nn.sigmoid,
    tanh = tf.nn.tanh,
    elu = tf.nn.elu,
    softplus = tf.nn.softplus,
  }
  local fn = supported[activation]
  if fn == nil then
    error("noisy_dense activation must be linear, relu, relu6, sigmoid, " ..
      "tanh, elu, or softplus", 3)
  end
  return fn(graph, input, name, dtype)
end

-- Noisy dense layer.  The layer learns a mean kernel/bias plus a sigma
-- kernel/bias and injects factorised (or independent) Gaussian noise between
-- them, following tfa.layers.NoisyDense.
function Layers.noisy_dense(graph, input, units, options)
  options = options or {}
  if type(units) ~= "number" or units % 1 ~= 0 or units < 1 then
    error("noisy_dense requires units to be a positive integer", 3)
  end
  local input_shape = options.input_shape or input.shape
  if type(input_shape) ~= "table" or #input_shape < 2 then
    error("noisy_dense requires options.input_shape with rank >= 2", 3)
  end
  local input_dim = input_shape[#input_shape]
  if type(input_dim) ~= "number" or input_dim % 1 ~= 0 or input_dim < 1 then
    error("noisy_dense requires a defined input feature dimension", 3)
  end
  local name = options.name or "noisy_dense"
  local dtype = options.dtype or input.dtype or "float32"
  local sigma = options.sigma
  if sigma == nil then sigma = 0.5 end
  if type(sigma) ~= "number" or sigma ~= sigma or sigma < 0 or sigma > 1 then
    error("noisy_dense sigma must be a number in [0, 1]", 3)
  end
  local factorised = options.use_factorised ~= false
  local use_bias = options.use_bias ~= false
  local mu_bound, sigma_init
  if factorised then
    mu_bound = 1 / math.sqrt(input_dim)
    sigma_init = sigma / math.sqrt(input_dim)
  else
    mu_bound = math.sqrt(3 / input_dim)
    sigma_init = 0.017
  end
  local kernel_shape = { input_dim, units }
  local bias_shape = { units }
  local mu_kernel = tf.variable(graph, random_array(kernel_shape, mu_bound),
    name .. "/mu_kernel", { dtype = dtype })
  local sigma_kernel = tf.variable(graph,
    filled(kernel_shape, sigma_init), name .. "/sigma_kernel",
    { dtype = dtype })
  local variables = { mu_kernel, sigma_kernel }
  local initializers = { mu_kernel.initializer, sigma_kernel.initializer }
  local mu_bias, sigma_bias, bias
  if use_bias then
    mu_bias = tf.variable(graph, random_array(bias_shape, mu_bound),
      name .. "/mu_bias", { dtype = dtype })
    sigma_bias = tf.variable(graph, filled(bias_shape, sigma_init),
      name .. "/sigma_bias", { dtype = dtype })
    variables[#variables + 1] = mu_bias
    variables[#variables + 1] = sigma_bias
    initializers[#initializers + 1] = mu_bias.initializer
    initializers[#initializers + 1] = sigma_bias.initializer
  end
  local eps_kernel, eps_bias
  if options.remove_noise == true then
    eps_kernel = tf.constant(graph, filled(kernel_shape, 0),
      name .. "/eps_kernel", dtype)
    eps_bias = tf.constant(graph, filled(bias_shape, 0),
      name .. "/eps_bias", dtype)
  elseif options.noise_kernel ~= nil or options.noise_bias ~= nil then
    if options.noise_kernel == nil or
        (use_bias and options.noise_bias == nil) then
      error("noisy_dense requires both noise_kernel and noise_bias", 3)
    end
    eps_kernel = options.noise_kernel
    eps_bias = options.noise_bias or tf.constant(graph, filled(bias_shape, 0),
      name .. "/eps_bias_zero", dtype)
  elseif factorised then
    local incoming = scaled_noise(graph, { input_dim, 1 },
      name .. "/incoming_noise", dtype, options.seed, options.seed2)
    local outgoing = scaled_noise(graph, { 1, units },
      name .. "/outgoing_noise", dtype, options.seed, options.seed2)
    eps_kernel = tf.matmul(graph, incoming, outgoing, name .. "/eps_kernel",
      dtype)
    eps_bias = tf.squeeze(graph, outgoing, 0, name .. "/eps_bias", dtype)
  else
    eps_kernel = random_standard_normal(graph, kernel_shape,
      name .. "/eps_kernel", dtype, options.seed, options.seed2)
    eps_bias = random_standard_normal(graph, bias_shape, name .. "/eps_bias",
      dtype, options.seed, options.seed2)
  end
  local kernel = tf.add(graph, mu_kernel.read, tf.mul(graph,
    sigma_kernel.read, eps_kernel, name .. "/sigma_eps_kernel", dtype),
    name .. "/kernel", dtype)
  if use_bias then
    bias = tf.add(graph, mu_bias.read, tf.mul(graph, sigma_bias.read,
      eps_bias, name .. "/sigma_eps_bias", dtype), name .. "/bias", dtype)
  end
  local flat_input = tf.reshape(graph, input, { -1, input_dim },
    name .. "/flat_input", dtype)
  local output = tf.matmul(graph, flat_input, kernel, name .. "/matmul", dtype)
  if use_bias then
    output = tf.add(graph, output, bias, name .. "/biased", dtype)
  end
  local output_shape = {}
  for index = 1, #input_shape do output_shape[index] = input_shape[index] end
  output_shape[1] = -1
  output_shape[#output_shape] = units
  output = tf.reshape(graph, output, output_shape, name .. "/output", dtype)
  output = noisy_dense_activation(graph, output, options.activation,
    name .. "/activation", dtype)
  return {
    output = output,
    kernel = kernel,
    bias = bias,
    eps_kernel = eps_kernel,
    eps_bias = eps_bias,
    mu_kernel = mu_kernel,
    sigma_kernel = sigma_kernel,
    mu_bias = mu_bias,
    sigma_bias = sigma_bias,
    variables = variables,
    initializers = initializers,
    units = units,
    input_dim = input_dim,
    use_factorised = factorised,
    use_bias = use_bias,
  }
end

Layers.NoisyDense = Layers.noisy_dense

local function rsqrt(graph, input, name, dtype)
  local desc = graph:new_operation("Rsqrt", name)
  desc:add_input(input)
  desc:set_type("T", dtype)
  return desc:finish()
end

-- Project vectors into the Poincare ball with norm <= 1 - epsilon, following
-- tfa.layers.PoincareNormalize.  Each slice along options.axis is normalized
-- independently; slices already inside the ball are unchanged.
function Layers.poincare_normalize(graph, input, options)
  options = options or {}
  local name = options.name or "poincare_normalize"
  local dtype = options.dtype or input.dtype or "float32"
  local epsilon = options.epsilon
  if epsilon == nil then epsilon = 1e-5 end
  if type(epsilon) ~= "number" or epsilon ~= epsilon or epsilon <= 0 or
      epsilon >= 1 or epsilon == math.huge then
    error("poincare_normalize epsilon must be in (0, 1)", 3)
  end
  local axis = options.axis
  if axis == nil then axis = 1 end
  if type(axis) == "number" then axis = { axis } end
  if type(axis) ~= "table" or #axis == 0 then
    error("poincare_normalize axis must be an integer or list", 3)
  end
  for _, value in ipairs(axis) do
    if type(value) ~= "number" or value % 1 ~= 0 then
      error("poincare_normalize axis entries must be integers", 3)
    end
  end
  local squared = tf.square(graph, input, name .. "/squared", dtype)
  local square_sum = tf.reduce_sum(graph, squared, axis,
    name .. "/square_sum", dtype, true)
  local inverse_norm = rsqrt(graph, square_sum, name .. "/inverse_norm", dtype)
  local boundary = tf.constant(graph, 1 - epsilon, name .. "/boundary", dtype)
  local scale = Ops.minimum(graph, tf.mul(graph, inverse_norm, boundary,
    name .. "/scaled_inverse_norm", dtype), tf.constant(graph, 1,
    name .. "/one", dtype), name .. "/scale", dtype)
  return {
    output = tf.mul(graph, input, scale, name .. "/output", dtype),
    axis = axis,
    epsilon = epsilon,
  }
end

Layers.PoincareNormalize = Layers.poincare_normalize

-- Thresholded Linear Unit, following tfa.layers.TLU.  Each non-batch
-- coordinate has a learned threshold tau; the optional affine form adds a
-- learned slope alpha to the threshold.
function Layers.tlu(graph, input, options)
  options = options or {}
  local input_shape = options.input_shape or input.shape
  if type(input_shape) ~= "table" or #input_shape < 2 then
    error("tlu requires options.input_shape with rank >= 2", 3)
  end
  local name = options.name or "tlu"
  local dtype = options.dtype or input.dtype or "float32"
  local parameter_shape = {}
  for index = 2, #input_shape do
    parameter_shape[#parameter_shape + 1] = input_shape[index]
  end
  local tau_value = options.tau or 0
  local alpha_value = options.alpha or 0
  for _, value in ipairs({ tau_value, alpha_value }) do
    if type(value) ~= "number" or value ~= value or value == math.huge or
        value == -math.huge then
      error("tlu tau and alpha must be finite numbers", 3)
    end
  end
  local tau = tf.variable(graph, filled(parameter_shape, tau_value),
    name .. "/tau", { dtype = dtype })
  local variables = { tau }
  local initializers = { tau.initializer }
  local alpha
  if options.affine == true then
    alpha = tf.variable(graph, filled(parameter_shape, alpha_value),
      name .. "/alpha", { dtype = dtype })
    variables[#variables + 1] = alpha
    initializers[#initializers + 1] = alpha.initializer
  end
  local threshold = tau.read
  if alpha ~= nil then
    threshold = tf.add(graph, threshold, tf.mul(graph, alpha.read, input,
      name .. "/alpha_term", dtype), name .. "/affine_threshold", dtype)
  end
  return {
    output = Ops.maximum(graph, input, threshold, name .. "/output", dtype),
    tau = tau,
    alpha = alpha,
    variables = variables,
    initializers = initializers,
  }
end

Layers.TLU = Layers.tlu

-- Polynomial crossing layer from tfa.layers.PolynomialCrossing.  It takes the
-- original input x0 and the previous crossing x and returns
-- x0 * (W x + diag_scale * x) + bias + x.
function Layers.polynomial_crossing(graph, x0, x, options)
  options = options or {}
  local input_shape = options.input_shape or x0.shape
  if type(input_shape) ~= "table" then
    error("polynomial_crossing requires options.input_shape", 3)
  end
  local input_dim = input_shape[#input_shape]
  if type(input_dim) ~= "number" or input_dim % 1 ~= 0 or input_dim < 1 then
    error("polynomial_crossing requires a defined input dimension", 3)
  end
  local name = options.name or "polynomial_crossing"
  local dtype = options.dtype or x0.dtype or "float32"
  local projection_dim = options.projection_dim
  local diag_scale = options.diag_scale or 0
  if type(diag_scale) ~= "number" or diag_scale ~= diag_scale or
      diag_scale < 0 or diag_scale == math.huge then
    error("polynomial_crossing diag_scale must be finite and non-negative", 3)
  end
  local bound = 1 / math.sqrt(input_dim)
  local variables, initializers = {}, {}
  local kernel, kernel_u, kernel_v
  if projection_dim == nil then
    kernel = tf.variable(graph, random_array({ input_dim, input_dim }, bound),
      name .. "/kernel", { dtype = dtype })
    variables[#variables + 1] = kernel
    initializers[#initializers + 1] = kernel.initializer
  else
    if type(projection_dim) ~= "number" or projection_dim % 1 ~= 0 or
        projection_dim < 1 or projection_dim > input_dim / 2 then
      error("polynomial_crossing projection_dim must be a positive integer " ..
        "at most half the input dimension", 3)
    end
    kernel_u = tf.variable(graph,
      random_array({ input_dim, projection_dim }, bound),
      name .. "/kernel_u", { dtype = dtype })
    kernel_v = tf.variable(graph,
      random_array({ projection_dim, input_dim }, bound),
      name .. "/kernel_v", { dtype = dtype })
    variables[#variables + 1] = kernel_u
    variables[#variables + 1] = kernel_v
    initializers[#initializers + 1] = kernel_u.initializer
    initializers[#initializers + 1] = kernel_v.initializer
  end
  local bias
  if options.use_bias ~= false then
    bias = tf.variable(graph, filled({ input_dim }, 0), name .. "/bias",
      { dtype = dtype })
    variables[#variables + 1] = bias
    initializers[#initializers + 1] = bias.initializer
  end
  local product
  if projection_dim == nil then
    product = tf.matmul(graph, x, kernel.read, name .. "/product", dtype)
  else
    local projected = tf.matmul(graph, x, kernel_u.read,
      name .. "/projected", dtype)
    product = tf.matmul(graph, projected, kernel_v.read,
      name .. "/product", dtype)
  end
  if diag_scale ~= 0 then
    product = tf.add(graph, product, tf.mul(graph, x,
      tf.constant(graph, diag_scale, name .. "/diag_scale", dtype),
      name .. "/diagonal", dtype), name .. "/scaled_product", dtype)
  end
  local output = tf.add(graph, tf.mul(graph, x0, product,
    name .. "/crossed", dtype), x, name .. "/residual", dtype)
  if bias ~= nil then
    output = tf.add(graph, output, bias.read, name .. "/output", dtype)
  end
  return {
    output = output,
    kernel = kernel,
    kernel_u = kernel_u,
    kernel_v = kernel_v,
    bias = bias,
    variables = variables,
    initializers = initializers,
  }
end

Layers.PolynomialCrossing = Layers.polynomial_crossing

local function dynamic_reshape(graph, input, shape, name, dtype)
  local desc = graph:new_operation("Reshape", name)
  desc:add_input(input)
  desc:add_input(shape)
  desc:set_type("T", dtype)
  desc:set_type("Tshape", "int32")
  return desc:finish()
end

local function sparsemax_last(graph, logits, name, dtype, rank)
  local shape = tf.shape(graph, logits, name .. "/shape", dtype)
  local prefix = tf.slice(graph, shape, { 0 }, { rank - 1 },
    name .. "/prefix", "int32")
  local observations = tf.reduce_prod(graph, prefix, { 0 },
    name .. "/observations", "int32", false)
  local dimensions = tf.gather(graph, shape,
    tf.constant(graph, rank - 1, name .. "/dimension_index", "int32"), 0,
    name .. "/dimensions", "int32")
  local matrix_shape = tf.stack(graph, { observations, dimensions }, 0,
    name .. "/matrix_shape", "int32")
  local z = dynamic_reshape(graph, logits, matrix_shape, name .. "/matrix",
    dtype)
  local top_desc = graph:new_operation("TopKV2", name .. "/top_k")
  top_desc:add_input(z)
  top_desc:add_input(dimensions)
  top_desc:set_type("T", dtype)
  top_desc:set_bool("sorted", true)
  local top = top_desc:finish()
  local z_sorted = { oper = top.oper, index = 0, name = name .. "/top_k:0",
    dtype = dtype }
  local cumsum_desc = graph:new_operation("Cumsum", name .. "/cumsum")
  cumsum_desc:add_input(z_sorted)
  cumsum_desc:add_input(tf.constant(graph, -1, name .. "/cumsum_axis",
    "int32"))
  cumsum_desc:set_type("T", dtype)
  cumsum_desc:set_type("Tidx", "int32")
  cumsum_desc:set_bool("exclusive", false)
  cumsum_desc:set_bool("reverse", false)
  local z_cumsum = cumsum_desc:finish()
  local range_desc = graph:new_operation("Range", name .. "/range")
  range_desc:add_input(tf.constant(graph, 0, name .. "/range_start", "int32"))
  range_desc:add_input(dimensions)
  range_desc:add_input(tf.constant(graph, 1, name .. "/range_delta", "int32"))
  range_desc:set_type("Tidx", "int32")
  local indices = range_desc:finish()
  local k = tf.add(graph, tf.cast(graph, indices, "int32", dtype,
    name .. "/range_float"), tf.constant(graph, 1, name .. "/one", dtype),
    name .. "/k", dtype)
  local one_plus = tf.add(graph, tf.constant(graph, 1, name .. "/one_plus",
    dtype), tf.mul(graph, k, z_sorted, name .. "/weighted_sorted", dtype),
    name .. "/one_plus_weighted", dtype)
  local comparison = Ops.greater(graph, one_plus, z_cumsum,
    name .. "/comparison", dtype)
  local comparison_int = tf.cast(graph, comparison, "bool", "int32",
    name .. "/comparison_int")
  local k_z = tf.reduce_sum(graph, comparison_int, { 1 }, name .. "/k_z",
    "int32", false)
  local k_safe = Ops.maximum(graph, k_z,
    tf.constant(graph, 1, name .. "/k_one", "int32"), name .. "/k_safe",
    "int32")
  local observation_range_desc = graph:new_operation("Range",
    name .. "/observation_range")
  observation_range_desc:add_input(tf.constant(graph, 0,
    name .. "/observation_start", "int32"))
  observation_range_desc:add_input(observations)
  observation_range_desc:add_input(tf.constant(graph, 1,
    name .. "/observation_delta", "int32"))
  observation_range_desc:set_type("Tidx", "int32")
  local observation_indices = observation_range_desc:finish()
  local gather_indices = tf.stack(graph, {
    observation_indices,
    tf.sub(graph, k_safe, tf.constant(graph, 1, name .. "/index_one",
      "int32"), name .. "/k_index", "int32"),
  }, 1, name .. "/gather_indices", "int32")
  local gather_desc = graph:new_operation("GatherNd", name .. "/gather")
  gather_desc:add_input(z_cumsum)
  gather_desc:add_input(gather_indices)
  gather_desc:set_type("Tparams", dtype)
  gather_desc:set_type("Tindices", "int32")
  local tau_sum = gather_desc:finish()
  local tau = tf.div(graph, tf.sub(graph, tau_sum,
    tf.constant(graph, 1, name .. "/tau_one", dtype), name .. "/tau_numerator",
    dtype), tf.cast(graph, k_z, "int32", dtype, name .. "/k_z_float"),
    name .. "/tau", dtype)
  local shifted = tf.sub(graph, z, tf.expand_dims(graph, tau, 1,
    name .. "/tau_expanded", dtype), name .. "/shifted", dtype)
  local sparse = Ops.maximum(graph, shifted,
    tf.constant(graph, 0, name .. "/sparse_zero", dtype), name .. "/sparse",
    dtype)
  local original_shape = tf.shape(graph, logits, name .. "/original_shape",
    dtype)
  return dynamic_reshape(graph, sparse, original_shape, name .. "/output",
    dtype)
end

-- Sparsemax activation, following tfa.layers.Sparsemax.  The projection is
-- applied along options.axis (default last) using a stable sort/cumsum.
function Layers.sparsemax(graph, input, options)
  options = options or {}
  local name = options.name or "sparsemax"
  local dtype = options.dtype or input.dtype or "float32"
  if dtype ~= "float32" and dtype ~= "float64" then
    error(name .. " requires a float32 or float64 input", 3)
  end
  local input_shape = options.input_shape or input.shape
  local rank = type(input_shape) == "table" and #input_shape or 2
  if rank < 2 then
    error(name .. " requires an input of rank at least 2", 3)
  end
  local axis = options.axis or -1
  if type(axis) ~= "number" or axis % 1 ~= 0 then
    error(name .. " axis must be an integer", 3)
  end
  if axis < 0 then axis = axis + rank end
  if axis < 0 or axis >= rank then
    error(name .. " axis is out of range", 3)
  end
  if axis == rank - 1 then
    return {
      output = sparsemax_last(graph, input, name, dtype, rank),
      axis = axis,
    }
  end
  local perm = {}
  for index = 0, rank - 1 do
    if index ~= axis then perm[#perm + 1] = index end
  end
  perm[#perm + 1] = axis
  local inverse = {}
  for index = 1, rank do inverse[perm[index] + 1] = index - 1 end
  local transposed = tf.transpose(graph, input, perm, name .. "/transpose",
    dtype)
  local output = sparsemax_last(graph, transposed, name .. "/projected", dtype,
    rank)
  output = tf.transpose(graph, output, inverse, name .. "/output", dtype)
  return { output = output, axis = axis }
end

Layers.Sparsemax = Layers.sparsemax

local function pooling_2d(graph, input, output_size, options, reduce,
    reducer_name)
  options = options or {}
  local name = options.name or reducer_name .. "_pooling_2d"
  local dtype = options.dtype or input.dtype or "float32"
  local data_format = options.data_format or "channels_last"
  if data_format ~= "channels_last" and data_format ~= "channels_first" then
    error(name .. " data_format must be channels_last or channels_first", 3)
  end
  local bins = output_size
  if type(bins) == "number" then bins = { bins, bins } end
  if type(bins) ~= "table" or #bins ~= 2 then
    error(name .. " output_size must be an integer or {rows, cols}", 3)
  end
  for _, value in ipairs(bins) do
    if type(value) ~= "number" or value % 1 ~= 0 or value < 1 then
      error(name .. " output_size entries must be positive integers", 3)
    end
  end
  local input_shape = options.input_shape or input.shape
  if type(input_shape) == "table" then
    if #input_shape ~= 4 then
      error(name .. " requires rank-4 inputs", 3)
    end
    local height, width
    if data_format == "channels_last" then
      height, width = input_shape[2], input_shape[3]
    else
      height, width = input_shape[3], input_shape[4]
    end
    if height % bins[1] ~= 0 or width % bins[2] ~= 0 then
      error(name .. " input height and width must be divisible by output_size",
        3)
    end
  end
  local height_axis = data_format == "channels_last" and 1 or 2
  local width_axis = data_format == "channels_last" and 2 or 3
  local row_splits = tf.split(graph, input, bins[1], height_axis,
    name .. "/rows", dtype)
  local rows = tf.stack(graph, row_splits, height_axis, name .. "/row_stack",
    dtype)
  local column_splits = tf.split(graph, rows, bins[2], width_axis + 1,
    name .. "/columns", dtype)
  local cells = tf.stack(graph, column_splits, width_axis + 1,
    name .. "/column_stack", dtype)
  local axes
  if data_format == "channels_last" then
    axes = { 2, 4 }
  else
    axes = { 3, 5 }
  end
  return reduce(graph, cells, axes, name .. "/output", dtype, false)
end

function Layers.adaptive_average_pooling_2d(graph, input, output_size, options)
  return pooling_2d(graph, input, output_size, options, tf.reduce_mean,
    "adaptive_average")
end

Layers.AdaptiveAveragePooling2D = Layers.adaptive_average_pooling_2d

function Layers.adaptive_max_pooling_2d(graph, input, output_size, options)
  return pooling_2d(graph, input, output_size, options, tf.reduce_max,
    "adaptive_max")
end

Layers.AdaptiveMaxPooling2D = Layers.adaptive_max_pooling_2d

local function pooling_1d(graph, input, output_size, options, reduce,
    reducer_name)
  options = options or {}
  local name = options.name or reducer_name .. "_pooling_1d"
  local dtype = options.dtype or input.dtype or "float32"
  local data_format = options.data_format or "channels_last"
  if data_format ~= "channels_last" and data_format ~= "channels_first" then
    error(name .. " data_format must be channels_last or channels_first", 3)
  end
  local bins = output_size
  if type(bins) ~= "number" or bins % 1 ~= 0 or bins < 1 then
    error(name .. " output_size must be a positive integer", 3)
  end
  local input_shape = options.input_shape or input.shape
  local length
  if type(input_shape) == "table" then
    if #input_shape ~= 3 then
      error(name .. " requires rank-3 inputs", 3)
    end
    length = data_format == "channels_last" and input_shape[2] or
      input_shape[3]
    if length % bins ~= 0 then
      error(name .. " input length must be divisible by output_size", 3)
    end
  end
  local length_axis = data_format == "channels_last" and 1 or 2
  local splits = tf.split(graph, input, bins, length_axis, name .. "/splits",
    dtype)
  local cells = tf.stack(graph, splits, length_axis, name .. "/stack", dtype)
  return reduce(graph, cells, { length_axis + 1 }, name .. "/output", dtype,
    false)
end

function Layers.adaptive_average_pooling_1d(graph, input, output_size, options)
  return pooling_1d(graph, input, output_size, options, tf.reduce_mean,
    "adaptive_average")
end

Layers.AdaptiveAveragePooling1D = Layers.adaptive_average_pooling_1d

function Layers.adaptive_max_pooling_1d(graph, input, output_size, options)
  return pooling_1d(graph, input, output_size, options, tf.reduce_max,
    "adaptive_max")
end

Layers.AdaptiveMaxPooling1D = Layers.adaptive_max_pooling_1d

local function pooling_3d(graph, input, output_size, options, reduce,
    reducer_name)
  options = options or {}
  local name = options.name or reducer_name .. "_pooling_3d"
  local dtype = options.dtype or input.dtype or "float32"
  local data_format = options.data_format or "channels_last"
  if data_format ~= "channels_last" and data_format ~= "channels_first" then
    error(name .. " data_format must be channels_last or channels_first", 3)
  end
  local bins = output_size
  if type(bins) == "number" then bins = { bins, bins, bins } end
  if type(bins) ~= "table" or #bins ~= 3 then
    error(name .. " output_size must be an integer or {depth, rows, cols}", 3)
  end
  for _, value in ipairs(bins) do
    if type(value) ~= "number" or value % 1 ~= 0 or value < 1 then
      error(name .. " output_size entries must be positive integers", 3)
    end
  end
  local input_shape = options.input_shape or input.shape
  if type(input_shape) == "table" then
    if #input_shape ~= 5 then
      error(name .. " requires rank-5 inputs", 3)
    end
    local depth, height, width
    if data_format == "channels_last" then
      depth, height, width = input_shape[2], input_shape[3], input_shape[4]
    else
      depth, height, width = input_shape[3], input_shape[4], input_shape[5]
    end
    if depth % bins[1] ~= 0 or height % bins[2] ~= 0 or
        width % bins[3] ~= 0 then
      error(name .. " spatial dimensions must be divisible by output_size", 3)
    end
  end
  local depth_axis = data_format == "channels_last" and 1 or 2
  local height_axis = data_format == "channels_last" and 2 or 3
  local width_axis = data_format == "channels_last" and 3 or 4
  local depth_parts = tf.split(graph, input, bins[1], depth_axis,
    name .. "/depth_splits", dtype)
  local depth_stack = tf.stack(graph, depth_parts, depth_axis,
    name .. "/depth_stack", dtype)
  local height_parts = tf.split(graph, depth_stack, bins[2], height_axis + 1,
    name .. "/height_splits", dtype)
  local height_stack = tf.stack(graph, height_parts, height_axis + 1,
    name .. "/height_stack", dtype)
  local width_parts = tf.split(graph, height_stack, bins[3], width_axis + 2,
    name .. "/width_splits", dtype)
  local cells = tf.stack(graph, width_parts, width_axis + 2,
    name .. "/width_stack", dtype)
  local axes = data_format == "channels_last" and { 2, 4, 6 } or { 3, 5, 7 }
  return reduce(graph, cells, axes, name .. "/output", dtype, false)
end

function Layers.adaptive_average_pooling_3d(graph, input, output_size, options)
  return pooling_3d(graph, input, output_size, options, tf.reduce_mean,
    "adaptive_average")
end

Layers.AdaptiveAveragePooling3D = Layers.adaptive_average_pooling_3d

function Layers.adaptive_max_pooling_3d(graph, input, output_size, options)
  return pooling_3d(graph, input, output_size, options, tf.reduce_max,
    "adaptive_max")
end

Layers.AdaptiveMaxPooling3D = Layers.adaptive_max_pooling_3d

local function scalar_dimension(graph, tensor, index, name, dtype)
  local shape = tf.shape(graph, tensor, name .. "/shape", dtype)
  return tf.gather(graph, shape,
    tf.constant(graph, index, name .. "/index", "int32"), 0, name, "int32")
end

-- NetVLAD aggregation layer, following tfa.layers.NetVLAD.
function Layers.netvlad(graph, input, num_clusters, options)
  options = options or {}
  if type(num_clusters) ~= "number" or num_clusters % 1 ~= 0 or
      num_clusters < 1 then
    error("netvlad requires a positive integer cluster count", 3)
  end
  local input_shape = options.input_shape or input.shape
  if type(input_shape) ~= "table" or #input_shape ~= 3 then
    error("netvlad requires options.input_shape with rank 3", 3)
  end
  local frames = input_shape[2]
  local feature_dim = input_shape[3]
  if type(frames) ~= "number" or type(feature_dim) ~= "number" or
      frames < 1 or feature_dim < 1 then
    error("netvlad requires static time and feature dimensions", 3)
  end
  local name = options.name or "netvlad"
  local dtype = options.dtype or input.dtype or "float32"
  local bound = 1 / math.sqrt(feature_dim)
  local kernel = tf.variable(graph,
    random_array({ feature_dim, num_clusters }, bound), name .. "/kernel",
    { dtype = dtype })
  local bias = tf.variable(graph, filled({ num_clusters }, 0),
    name .. "/bias", { dtype = dtype })
  local centers = tf.variable(graph,
    random_array({ 1, feature_dim, num_clusters }, bound),
    name .. "/cluster_centers", { dtype = dtype })
  local variables = { kernel, bias, centers }
  local initializers = { kernel.initializer, bias.initializer,
    centers.initializer }
  local flat = tf.reshape(graph, input, { -1, feature_dim },
    name .. "/flat", dtype)
  local assignment = tf.nn.softmax(graph, tf.add(graph, tf.matmul(graph, flat,
    kernel.read, name .. "/assignment_logits", dtype), bias.read,
    name .. "/assignment_biased", dtype), name .. "/assignment", dtype)
  assignment = tf.reshape(graph, assignment, { -1, frames, num_clusters },
    name .. "/assignment_reshaped", dtype)
  local assignment_transposed = tf.transpose(graph, assignment, { 0, 2, 1 },
    name .. "/assignment_transposed", dtype)
  local weighted = tf.batch_matmul(graph, assignment_transposed, input,
    name .. "/weighted_frames", dtype)
  local b = tf.transpose(graph, weighted, { 0, 2, 1 },
    name .. "/weighted_transposed", dtype)
  local a_sum = tf.reduce_sum(graph, assignment, { 1 },
    name .. "/assignment_sum", dtype, true)
  local a = tf.mul(graph, a_sum, centers.read, name .. "/center_sum", dtype)
  local vlad = tf.sub(graph, b, a, name .. "/vlad", dtype)
  local squared = tf.square(graph, vlad, name .. "/squared", dtype)
  local norm = tf.sqrt(graph, tf.reduce_sum(graph, squared, { 1 },
    name .. "/feature_norm_squared", dtype, true),
    name .. "/feature_norm", dtype)
  vlad = tf.div(graph, vlad, Ops.maximum(graph, norm,
    tf.constant(graph, 1e-12, name .. "/epsilon", dtype),
    name .. "/safe_norm", dtype), name .. "/feature_normalized", dtype)
  vlad = tf.reshape(graph, vlad, { -1, feature_dim * num_clusters },
    name .. "/flattened", dtype)
  local flat_squared = tf.square(graph, vlad, name .. "/flat_squared", dtype)
  local flat_norm = tf.sqrt(graph, tf.reduce_sum(graph, flat_squared, { 1 },
    name .. "/flat_norm_squared", dtype, true), name .. "/flat_norm", dtype)
  local output = tf.div(graph, vlad, Ops.maximum(graph, flat_norm,
    tf.constant(graph, 1e-12, name .. "/flat_epsilon", dtype),
    name .. "/flat_safe_norm", dtype), name, dtype)
  return {
    output = output,
    kernel = kernel,
    bias = bias,
    cluster_centers = centers,
    variables = variables,
    initializers = initializers,
  }
end

Layers.NetVLAD = Layers.netvlad

local function int_range(graph, limit, name)
  local desc = graph:new_operation("Range", name)
  desc:add_input(tf.constant(graph, 0, name .. "/start", "int32"))
  desc:add_input(limit)
  desc:add_input(tf.constant(graph, 1, name .. "/delta", "int32"))
  desc:set_type("Tidx", "int32")
  return desc:finish()
end

local function floor_div(graph, a, b, name)
  local desc = graph:new_operation("FloorDiv", name)
  desc:add_input(a)
  desc:add_input(b)
  desc:set_type("T", "int32")
  return desc:finish()
end

local function int_mod(graph, a, b, name)
  local desc = graph:new_operation("FloorMod", name)
  desc:add_input(a)
  desc:add_input(b)
  desc:set_type("T", "int32")
  return desc:finish()
end

-- MaxUnpooling2D, following tfa.layers.MaxUnpooling2D.  `mask` is the argmax
-- output of a matching max-pooling operation.
function Layers.max_unpooling_2d(graph, updates, mask, options)
  options = options or {}
  local name = options.name or "max_unpooling_2d"
  local dtype = options.dtype or updates.dtype or "float32"
  local pool_size = options.pool_size or { 2, 2 }
  local strides = options.strides or { 2, 2 }
  if type(pool_size) == "number" then pool_size = { pool_size, pool_size } end
  if type(strides) == "number" then strides = { strides, strides } end
  for _, pair in ipairs({ pool_size, strides }) do
    if type(pair) ~= "table" or #pair ~= 2 then
      error(name .. " pool_size and strides must be pairs", 3)
    end
    for _, value in ipairs(pair) do
      if type(value) ~= "number" or value % 1 ~= 0 or value < 1 then
        error(name .. " pool_size and strides must be positive integers", 3)
      end
    end
  end
  local padding = string.upper(options.padding or "SAME")
  if padding ~= "SAME" and padding ~= "VALID" then
    error(name .. " padding must be 'SAME' or 'VALID'", 3)
  end
  local batch = scalar_dimension(graph, updates, 0, name .. "/batch", dtype)
  local height = scalar_dimension(graph, updates, 1, name .. "/height", dtype)
  local width = scalar_dimension(graph, updates, 2, name .. "/width", dtype)
  local channels = scalar_dimension(graph, updates, 3, name .. "/channels",
    dtype)
  local one_int = tf.constant(graph, 1, name .. "/one_int", "int32")
  local out_height, out_width
  if padding == "SAME" then
    out_height = tf.mul(graph, height, tf.constant(graph, strides[1],
      name .. "/stride_h", "int32"), name .. "/out_height", "int32")
    out_width = tf.mul(graph, width, tf.constant(graph, strides[2],
      name .. "/stride_w", "int32"), name .. "/out_width", "int32")
  else
    out_height = tf.add(graph, tf.mul(graph, tf.sub(graph, height, one_int,
      name .. "/height_minus_one", "int32"), tf.constant(graph, strides[1],
      name .. "/stride_h", "int32"), name .. "/valid_height", "int32"),
      tf.constant(graph, pool_size[1], name .. "/pool_h", "int32"),
      name .. "/out_height", "int32")
    out_width = tf.add(graph, tf.mul(graph, tf.sub(graph, width, one_int,
      name .. "/width_minus_one", "int32"), tf.constant(graph, strides[2],
      name .. "/stride_w", "int32"), name .. "/valid_width", "int32"),
      tf.constant(graph, pool_size[2], name .. "/pool_w", "int32"),
      name .. "/out_width", "int32")
  end
  local output_shape = pack_int32(graph, { batch, out_height, out_width,
    channels }, name .. "/output_shape")
  local mask_int = tf.cast(graph, mask, mask.dtype or "int32", "int32",
    name .. "/mask_int")
  local zero = tf.constant(graph, 0, name .. "/zero", "int32")
  local one_mask = tf.add(graph, tf.mul(graph, mask_int, zero,
    name .. "/mask_zero", "int32"), one_int, name .. "/one_mask", "int32")
  local batch_range = int_range(graph, batch, name .. "/batch_range")
  local batch_shape = pack_int32(graph, { batch, 1, 1, 1 },
    name .. "/batch_range_shape")
  batch_range = dynamic_reshape(graph, batch_range, batch_shape,
    name .. "/batch_range_reshaped", "int32")
  local batch_indices = tf.mul(graph, one_mask, batch_range,
    name .. "/batch_indices", "int32")
  local feature_range = int_range(graph, channels, name .. "/feature_range")
  local feature_indices = tf.mul(graph, one_mask, feature_range,
    name .. "/feature_indices", "int32")
  local area = tf.mul(graph, out_width, channels, name .. "/area", "int32")
  local y = floor_div(graph, mask_int, area, name .. "/y")
  local x = int_mod(graph, floor_div(graph, mask_int, channels, name .. "/x_raw"),
    out_width, name .. "/x")
  local indices = tf.stack(graph, { batch_indices, y, x, feature_indices }, 3,
    name .. "/indices_grid", "int32")
  local updates_size = tf.reshape(graph, updates, { -1 }, name .. "/updates_flat",
    dtype)
  local indices_flat = tf.reshape(graph, indices, { -1, 4 },
    name .. "/indices_flat", "int32")
  local desc = graph:new_operation("ScatterNd", name .. "/scatter")
  desc:add_input(indices_flat)
  desc:add_input(updates_size)
  desc:add_input(output_shape)
  desc:set_type("T", dtype)
  desc:set_type("Tindices", "int32")
  return desc:finish()
end

Layers.MaxUnpooling2D = Layers.max_unpooling_2d

-- Snake layer with a trainable scalar frequency, following tfa.layers.Snake.
function Layers.snake(graph, input, options)
  options = options or {}
  local name = options.name or "snake"
  local dtype = options.dtype or input.dtype or "float32"
  local initial = options.frequency or 1
  if type(initial) ~= "number" or initial ~= initial or initial <= 0 or
      initial == math.huge then
    error("snake frequency must be a finite positive number", 3)
  end
  local Activations = require("tensorflow_addons.activations")
  local frequency = tf.variable(graph, initial, name .. "/frequency",
    { dtype = dtype })
  return {
    output = Activations.snake(graph, input, frequency.read,
      name .. "/output", dtype),
    frequency = frequency,
    variables = { frequency },
    initializers = { frequency.initializer },
  }
end

Layers.Snake = Layers.snake

-- GELU activation layer using the tanh approximation.
function Layers.gelu(graph, input, options)
  options = options or {}
  local Activations = require("tensorflow_addons.activations")
  local name = options.name or "gelu"
  return Activations.gelu(graph, input, name,
    options.dtype or input.dtype or "float32")
end

Layers.GELU = Layers.gelu

local function slice_4d(graph, input, begin_values, size_values, name, dtype)
  local desc = graph:new_operation("Slice", name)
  desc:add_input(input)
  desc:add_input(pack_int32(graph, begin_values, name .. "/begin"))
  desc:add_input(pack_int32(graph, size_values, name .. "/size"))
  desc:set_type("T", dtype)
  desc:set_type("Index", "int32")
  return desc:finish()
end

local function spatial_bin(graph, input, bin, data_format, name, dtype)
  local shape = tf.shape(graph, input, name .. "/shape", dtype)
  local function dim(index, label)
    return tf.gather(graph, shape,
      tf.constant(graph, index, name .. "/" .. label .. "_index", "int32"), 0,
      name .. "/" .. label, "int32")
  end
  local batch = dim(0, "batch")
  local channels = dim(data_format == "channels_last" and 3 or 1, "channels")
  local height = dim(data_format == "channels_last" and 1 or 2, "height")
  local width = dim(data_format == "channels_last" and 2 or 3, "width")
  local overflow_h = int_mod(graph, height, tf.constant(graph, bin[1],
    name .. "/bin_h", "int32"), name .. "/overflow_h")
  local overflow_w = int_mod(graph, width, tf.constant(graph, bin[2],
    name .. "/bin_w", "int32"), name .. "/overflow_w")
  local new_h = tf.sub(graph, height, overflow_h, name .. "/new_h", "int32")
  local new_w = tf.sub(graph, width, overflow_w, name .. "/new_w", "int32")
  local cropped
  if data_format == "channels_last" then
    cropped = slice_4d(graph, input, { 0, 0, 0, 0 },
      { batch, new_h, new_w, channels }, name .. "/crop", dtype)
  else
    cropped = slice_4d(graph, input, { 0, 0, 0, 0 },
      { batch, channels, new_h, new_w }, name .. "/crop", dtype)
  end
  local pooled = Layers.adaptive_average_pooling_2d(graph, cropped, bin,
    { data_format = data_format, name = name .. "/pool" })
  local product = bin[1] * bin[2]
  local target = pack_int32(graph, data_format == "channels_last" and
    { batch, product, channels } or { batch, channels, product },
    name .. "/target")
  return dynamic_reshape(graph, pooled, target, name .. "/bin", dtype)
end

-- Spatial pyramid pooling over adaptive average bins, following
-- tfa.layers.SpatialPyramidPooling2D.
function Layers.spatial_pyramid_pooling_2d(graph, input, bins, options)
  options = options or {}
  local name = options.name or "spatial_pyramid_pooling_2d"
  local dtype = options.dtype or input.dtype or "float32"
  local data_format = options.data_format or "channels_last"
  if data_format ~= "channels_last" and data_format ~= "channels_first" then
    error(name .. " data_format must be channels_last or channels_first", 3)
  end
  if type(bins) ~= "table" or #bins == 0 then
    error(name .. " requires a non-empty bins list", 3)
  end
  local normalized = {}
  for index, value in ipairs(bins) do
    local pair = value
    if type(value) == "number" then pair = { value, value } end
    if type(pair) ~= "table" or #pair ~= 2 then
      error(name .. " bins must be integers or {rows, cols} pairs", 3)
    end
    for _, entry in ipairs(pair) do
      if type(entry) ~= "number" or entry % 1 ~= 0 or entry < 1 then
        error(name .. " bin entries must be positive integers", 3)
      end
    end
    normalized[index] = pair
  end
  local outputs = {}
  for index, pair in ipairs(normalized) do
    outputs[index] = spatial_bin(graph, input, pair, data_format,
      name .. "/bin_" .. index, dtype)
  end
  return tf.concat(graph, outputs,
    data_format == "channels_last" and 1 or 2, name, dtype)
end

Layers.SpatialPyramidPooling2D = Layers.spatial_pyramid_pooling_2d

return Layers
