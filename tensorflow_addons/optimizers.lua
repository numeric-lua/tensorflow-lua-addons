local tf = require("tensorflow")
local Train = require("tensorflow.train")
local Ops = require("tensorflow_addons.ops")

local Optimizers = {}

local function zeros(shape, depth)
  depth = depth or 1
  if depth > #shape then return 0 end
  local result = {}
  for index = 1, shape[depth] do result[index] = zeros(shape, depth + 1) end
  return result
end

local function assign(graph, ref, value, name, dtype, controls)
  local desc = graph:new_operation("Assign", name)
  desc:add_input(ref)
  desc:add_input(value)
  desc:set_type("T", dtype)
  desc:set_bool("use_locking", true)
  for _, control in ipairs(controls or {}) do
    desc:add_control_input(control)
  end
  return desc:finish()
end

function Optimizers.LAMB(graph, loss, variables, options)
  options = options or {}
  local learning_rate = options.learning_rate or options.lr or 0.001
  local beta1 = options.beta1 or 0.9
  local beta2 = options.beta2 or 0.999
  local epsilon = options.epsilon or 1e-6
  local weight_decay = options.weight_decay or options.weight_decay_rate or 0
  local name = options.name or "LAMB"
  local refs = {}
  for index, variable in ipairs(variables) do
    refs[index] = variable.ref or variable
  end
  local gradients = tf.gradients(graph, { loss }, refs)
  local dtype = variables[1].dtype or "float32"
  local weight_decay_op = tf.constant(graph, weight_decay,
    name .. "/weight_decay", dtype)
  local lr = tf.constant(graph, learning_rate, name .. "/learning_rate", dtype)
  local b1 = tf.constant(graph, beta1, name .. "/beta1", dtype)
  local b2 = tf.constant(graph, beta2, name .. "/beta2", dtype)
  local one = tf.constant(graph, 1, name .. "/one", dtype)
  local eps = tf.constant(graph, epsilon, name .. "/epsilon", dtype)
  local beta1_power = Train.variable(graph, beta1, name .. "/beta1_power",
    { dtype = dtype })
  local beta2_power = Train.variable(graph, beta2, name .. "/beta2_power",
    { dtype = dtype })
  local initializers = { beta1_power.initializer, beta2_power.initializer }
  local state_variables = { beta1_power, beta2_power }
  local targets = {}

  local function sum_squares(value, label, suffix)
    local prefix = name .. "/" .. label .. "_" .. suffix
    local flat = tf.reshape(graph, value, { -1 }, prefix .. "_flat", dtype)
    return tf.reduce_sum(graph, tf.square(graph, flat, prefix .. "_square",
      dtype), { 0 }, prefix .. "_sum", dtype)
  end

  for index, variable in ipairs(variables) do
    local ref = variable.ref or variable
    local var_dtype = variable.dtype or dtype
    local shape = variable.shape or { }
    local m = Train.variable(graph, zeros(shape), name .. "/m_" .. index,
      { dtype = var_dtype })
    local v = Train.variable(graph, zeros(shape), name .. "/v_" .. index,
      { dtype = var_dtype })
    initializers[#initializers + 1] = m.initializer
    initializers[#initializers + 1] = v.initializer
    state_variables[#state_variables + 1] = m
    state_variables[#state_variables + 1] = v

    local gradient = gradients[index]
    local m_next = tf.add(graph, tf.mul(graph, m.read, b1,
      name .. "/m_decay_" .. index, var_dtype), tf.mul(graph, gradient,
      tf.sub(graph, one, b1, name .. "/one_minus_b1_" .. index, var_dtype),
      name .. "/m_gradient_" .. index, var_dtype), name .. "/m_next_" ..
      index, var_dtype)
    local v_next = tf.add(graph, tf.mul(graph, v.read, b2,
      name .. "/v_decay_" .. index, var_dtype), tf.mul(graph,
      tf.square(graph, gradient, name .. "/grad_square_" .. index, var_dtype),
      tf.sub(graph, one, b2, name .. "/one_minus_b2_" .. index, var_dtype),
      name .. "/v_gradient_" .. index, var_dtype), name .. "/v_next_" ..
      index, var_dtype)
    local m_assign = assign(graph, m.ref, m_next, name .. "/m_assign_" .. index,
      var_dtype)
    local v_assign = assign(graph, v.ref, v_next, name .. "/v_assign_" .. index,
      var_dtype)

    local one_minus_b1_power = tf.sub(graph, one, beta1_power.read,
      name .. "/one_minus_b1_power_" .. index, var_dtype)
    local one_minus_b2_power = tf.sub(graph, one, beta2_power.read,
      name .. "/one_minus_b2_power_" .. index, var_dtype)
    local m_hat = tf.div(graph, m_next, one_minus_b1_power,
      name .. "/m_hat_" .. index, var_dtype)
    local v_hat = tf.div(graph, v_next, one_minus_b2_power,
      name .. "/v_hat_" .. index, var_dtype)
    local update = tf.div(graph, m_hat, tf.add(graph,
      tf.sqrt(graph, v_hat, name .. "/v_sqrt_" .. index, var_dtype), eps,
      name .. "/v_epsilon_" .. index, var_dtype), name .. "/update_" ..
      index, var_dtype)

    local weight_norm = tf.sqrt(graph, tf.add(graph,
      sum_squares(variable.read, "weight_norm", index), eps,
      name .. "/weight_norm_epsilon_" .. index, var_dtype),
      name .. "/weight_norm_" .. index, var_dtype)
    local update_norm = tf.sqrt(graph, tf.add(graph,
      sum_squares(update, "update_norm", index), eps,
      name .. "/update_norm_epsilon_" .. index, var_dtype),
      name .. "/update_norm_" .. index, var_dtype)
    local trust = tf.div(graph, weight_norm, Ops.maximum(graph, update_norm,
      eps, name .. "/update_norm_floor_" .. index, var_dtype),
      name .. "/trust_" .. index, var_dtype)
    local scaled = tf.mul(graph, trust, update, name .. "/scaled_update_" ..
      index, var_dtype)
    local decay = tf.mul(graph, variable.read, weight_decay_op,
      name .. "/weight_decay_" .. index, var_dtype)
    local delta = tf.mul(graph, lr, tf.add(graph, scaled, decay,
      name .. "/update_and_decay_" .. index, var_dtype),
      name .. "/delta_" .. index, var_dtype)
    local next_weight = tf.sub(graph, variable.read, delta,
      name .. "/next_weight_" .. index, var_dtype)
    targets[#targets + 1] = assign(graph, ref, next_weight,
      name .. "/weight_assign_" .. index, var_dtype, { m_assign, v_assign })
  end

  if #targets > 0 then
    local next_b1 = tf.mul(graph, beta1_power.read, b1,
      name .. "/next_beta1", dtype)
    local next_b2 = tf.mul(graph, beta2_power.read, b2,
      name .. "/next_beta2", dtype)
    targets[#targets + 1] = assign(graph, beta1_power.ref, next_b1,
      name .. "/assign_beta1", dtype, targets)
    targets[#targets + 1] = assign(graph, beta2_power.ref, next_b2,
      name .. "/assign_beta2", dtype, targets)
  end

  return targets, gradients, {
    initializers = initializers, variables = state_variables,
    beta1_power = beta1_power, beta2_power = beta2_power,
  }
end

return Optimizers
