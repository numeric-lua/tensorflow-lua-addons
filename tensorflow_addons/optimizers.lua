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

local function identity(graph, input, name, dtype, controls)
  local desc = graph:new_operation("Identity", name)
  desc:add_input(input)
  desc:set_type("T", dtype)
  for _, control in ipairs(controls or {}) do
    desc:add_control_input(control)
  end
  return desc:finish()
end

-- Materialize a non-ref snapshot. Identity on a variable ref stays a ref, so
-- downstream ops may observe it at different times; adding zero forces a value
-- that is read after any requested control dependencies.
local function read_value(graph, ref, name, dtype, controls)
  local source = ref
  if controls ~= nil and #controls > 0 then
    source = identity(graph, ref, name .. "/read", dtype, controls)
  end
  local zero = tf.constant(graph, 0, name .. "/zero", dtype)
  return tf.add(graph, source, zero, name, dtype)
end

local function mod(graph, a, b, name, dtype)
  local desc = graph:new_operation("Mod", name)
  desc:add_input(a)
  desc:add_input(b)
  desc:set_type("T", dtype)
  return desc:finish()
end

local function equal(graph, a, b, name, dtype)
  local desc = graph:new_operation("Equal", name)
  desc:add_input(a)
  desc:add_input(b)
  desc:set_type("T", dtype)
  return desc:finish()
end

local function not_equal(graph, a, b, name, dtype)
  local desc = graph:new_operation("NotEqual", name)
  desc:add_input(a)
  desc:add_input(b)
  desc:set_type("T", dtype)
  return desc:finish()
end

local function graph_select(graph, condition, then_value, else_value, name,
    dtype)
  local desc = graph:new_operation("Select", name)
  desc:add_input(condition)
  desc:add_input(then_value)
  desc:add_input(else_value)
  desc:set_type("T", dtype)
  return desc:finish()
end

local function is_graph(value)
  return type(value) == "table" and type(value.new_operation) == "function"
end

local function is_callable(value)
  if type(value) == "function" then return true end
  if type(value) ~= "table" then return false end
  local meta = getmetatable(value)
  return meta ~= nil and type(meta.__call) == "function"
end

local function merge_tables(...)
  local result = {}
  for index = 1, select("#", ...) do
    local source = select(index, ...)
    for key, value in pairs(source or {}) do result[key] = value end
  end
  return result
end

local function append(destination, values)
  for _, value in ipairs(values or {}) do
    destination[#destination + 1] = value
  end
end

local function validate_period(value)
  if type(value) ~= "number" or value ~= value or value == math.huge or
      value == -math.huge or value ~= math.floor(value) or value < 1 then
    error("Lookahead sync_period must be a positive integer", 3)
  end
  return value
end

local function validate_slow_step_size(value)
  if type(value) ~= "number" or value ~= value or value == math.huge or
      value == -math.huge then
    error("Lookahead slow_step_size must be a finite number", 3)
  end
  return value
end

local function resolve_optimizer(base_optimizer)
  if type(base_optimizer) ~= "string" then return base_optimizer end
  local direct = Optimizers[base_optimizer]
  if type(direct) == "function" then return direct end
  local key = base_optimizer:lower()
  if key == "lamb" then return Optimizers.LAMB end
  if key == "novograd" then return Optimizers.NovoGrad end
  if key == "gradient_descent" or key == "gradientdescent" then
    return function(graph, loss, variables, options)
      options = options or {}
      return Train.gradient_descent(graph, loss, variables,
        options.learning_rate or options.lr or 0.001,
        options.name or "GradientDescent", options)
    end
  end
  return tf.train[key] or tf.train[base_optimizer]
end

local function split_options(options)
  local config, base_options = {}, {}
  local wrapper_keys = {
    sync_period = true, slow_step_size = true, name = true,
    base_name = true, base_options = true, base_optimizer = true,
    base = true, optimizer = true,
  }
  for key, value in pairs(options or {}) do
    if wrapper_keys[key] then
      config[key] = value
    else
      base_options[key] = value
    end
  end
  return config, base_options
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

-- Lazy Adam.  Unlike Adam, moment accumulators and variables are only updated
-- for tensor elements that receive a non-zero gradient.  This mirrors the
-- sparse-update savings of tfa.optimizers.LazyAdam while still accepting the
-- dense gradients returned by the public tensorflow.lua gradients API.
function Optimizers.LazyAdam(graph, loss, variables, options)
  options = options or {}
  if type(variables) ~= "table" or #variables == 0 then
    error("LazyAdam expects a non-empty variable list", 3)
  end
  local learning_rate = options.learning_rate or options.lr or 0.001
  local beta1 = options.beta1 or 0.9
  local beta2 = options.beta2 or 0.999
  local epsilon = options.epsilon or 1e-8
  local weight_decay = options.weight_decay or options.weight_decay_rate or
    options.wd or 0
  local name = options.name or "LazyAdam"
  local refs = {}
  for index, variable in ipairs(variables) do
    refs[index] = variable.ref or variable
  end
  local gradients = options.gradients or tf.gradients(graph, { loss }, refs)
  local dtype = variables[1].dtype or "float32"
  local lr = tf.constant(graph, learning_rate, name .. "/learning_rate", dtype)
  local b1 = tf.constant(graph, beta1, name .. "/beta1", dtype)
  local b2 = tf.constant(graph, beta2, name .. "/beta2", dtype)
  local eps = tf.constant(graph, epsilon, name .. "/epsilon", dtype)
  local zero = tf.constant(graph, 0, name .. "/zero", dtype)
  local one = tf.constant(graph, 1, name .. "/one", dtype)
  local weight_decay_op = tf.constant(graph, weight_decay,
    name .. "/weight_decay", dtype)
  local beta1_power = Train.variable(graph, beta1, name .. "/beta1_power",
    { dtype = dtype })
  local beta2_power = Train.variable(graph, beta2, name .. "/beta2_power",
    { dtype = dtype })
  local initializers = { beta1_power.initializer, beta2_power.initializer }
  local state_variables = { beta1_power, beta2_power }
  local targets = {}

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
    local mask = not_equal(graph, gradient, zero,
      name .. "/gradient_mask_" .. index, var_dtype)
    local m_candidate = tf.add(graph, tf.mul(graph, m.read, b1,
      name .. "/m_decay_" .. index, var_dtype), tf.mul(graph, gradient,
      tf.sub(graph, one, b1, name .. "/one_minus_b1_" .. index, var_dtype),
      name .. "/m_gradient_" .. index, var_dtype), name .. "/m_candidate_" ..
      index, var_dtype)
    local v_candidate = tf.add(graph, tf.mul(graph, v.read, b2,
      name .. "/v_decay_" .. index, var_dtype), tf.mul(graph,
      tf.square(graph, gradient, name .. "/grad_square_" .. index, var_dtype),
      tf.sub(graph, one, b2, name .. "/one_minus_b2_" .. index, var_dtype),
      name .. "/v_gradient_" .. index, var_dtype), name .. "/v_candidate_" ..
      index, var_dtype)
    local m_next = graph_select(graph, mask, m_candidate, m.read,
      name .. "/m_lazy_" .. index, var_dtype)
    local v_next = graph_select(graph, mask, v_candidate, v.read,
      name .. "/v_lazy_" .. index, var_dtype)
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
    local decay = tf.mul(graph, variable.read, weight_decay_op,
      name .. "/weight_decay_" .. index, var_dtype)
    local delta = tf.mul(graph, lr, tf.add(graph, update, decay,
      name .. "/update_and_decay_" .. index, var_dtype),
      name .. "/delta_" .. index, var_dtype)
    local candidate = tf.sub(graph, variable.read, delta,
      name .. "/candidate_" .. index, var_dtype)
    local next_weight = graph_select(graph, mask, candidate, variable.read,
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
    beta1_power = beta1_power, beta2_power = beta2_power, lazy = true,
  }
end

function Optimizers.NovoGrad(graph, loss, variables, options)
  options = options or {}
  local learning_rate = options.learning_rate or options.lr or 0.001
  local beta1 = options.beta1 or 0.9
  local beta2 = options.beta2 or 0.999
  local epsilon = options.epsilon or 1e-8
  local weight_decay = options.weight_decay or options.weight_decay_rate or 0
  local name = options.name or "NovoGrad"
  local refs = {}
  for index, variable in ipairs(variables) do
    refs[index] = variable.ref or variable
  end
  local gradients = tf.gradients(graph, { loss }, refs)
  local dtype = variables[1].dtype or "float32"
  local lr = tf.constant(graph, learning_rate, name .. "/learning_rate", dtype)
  local b1 = tf.constant(graph, beta1, name .. "/beta1", dtype)
  local b2 = tf.constant(graph, beta2, name .. "/beta2", dtype)
  local one = tf.constant(graph, 1, name .. "/one", dtype)
  local eps = tf.constant(graph, epsilon, name .. "/epsilon", dtype)
  local decay_op = tf.constant(graph, weight_decay, name .. "/weight_decay",
    dtype)
  local initializers, state_variables, targets = {}, {}, {}

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
    local v_next = tf.add(graph, tf.mul(graph, v.read, b2,
      name .. "/v_decay_" .. index, var_dtype), tf.mul(graph,
      tf.square(graph, gradient, name .. "/grad_square_" .. index, var_dtype),
      tf.sub(graph, one, b2, name .. "/one_minus_b2_" .. index, var_dtype),
      name .. "/v_gradient_" .. index, var_dtype), name .. "/v_next_" ..
      index, var_dtype)
    local v_denom = tf.add(graph, tf.sqrt(graph, v_next,
      name .. "/v_sqrt_" .. index, var_dtype), eps,
      name .. "/v_epsilon_" .. index, var_dtype)
    local normalized_gradient = tf.div(graph, tf.mul(graph, lr, gradient,
      name .. "/lr_gradient_" .. index, var_dtype), v_denom,
      name .. "/normalized_gradient_" .. index, var_dtype)
    local m_next = tf.add(graph, tf.mul(graph, m.read, b1,
      name .. "/m_decay_" .. index, var_dtype), normalized_gradient,
      name .. "/m_next_" .. index, var_dtype)
    local decay = tf.mul(graph, variable.read, decay_op,
      name .. "/weight_decay_" .. index, var_dtype)
    local next_weight = tf.sub(graph, variable.read, tf.add(graph, m_next,
      decay, name .. "/update_and_decay_" .. index, var_dtype),
      name .. "/next_weight_" .. index, var_dtype)
    local v_assign = assign(graph, v.ref, v_next,
      name .. "/v_assign_" .. index, var_dtype)
    local m_assign = assign(graph, m.ref, m_next,
      name .. "/m_assign_" .. index, var_dtype, { v_assign })
    targets[#targets + 1] = assign(graph, ref, next_weight,
      name .. "/weight_assign_" .. index, var_dtype, { m_assign })
  end
  return targets, gradients, { initializers = initializers,
    variables = state_variables }
end

function Optimizers.Lookahead(base_optimizer, options, direct_loss,
    direct_variables)
  if is_graph(base_optimizer) then
    local graph = base_optimizer
    local loss = options
    local variables = direct_loss
    local direct_options = direct_variables or {}
    local base = direct_options.base_optimizer or direct_options.base or
      direct_options.optimizer
    local optimizer = Optimizers.Lookahead(base, direct_options)
    return optimizer(graph, loss, variables, direct_options)
  end

  if type(base_optimizer) == "table" and not is_callable(base_optimizer) and
      options == nil then
    local config = base_optimizer
    base_optimizer = config.base_optimizer or config.base or config.optimizer
    options = config
  end

  base_optimizer = resolve_optimizer(base_optimizer)
  if not is_callable(base_optimizer) then
    error("Lookahead expects a callable base optimizer or optimizer name", 2)
  end

  local config, factory_base_options = split_options(options)
  config.sync_period = validate_period(config.sync_period or 5)
  config.slow_step_size = validate_slow_step_size(config.slow_step_size or 0.5)
  config.name = config.name or "Lookahead"
  config.base_name = config.base_name or (config.name .. "_base")
  local base_options = merge_tables(factory_base_options, config.base_options)
  if base_options.name == nil then base_options.name = config.base_name end

  return function(graph, loss, variables, call_options)
    call_options = call_options or {}
    local call_config, call_base_options = split_options(call_options)
    local sync_period = validate_period(call_config.sync_period or
      config.sync_period)
    local slow_step_size = validate_slow_step_size(call_config.slow_step_size or
      config.slow_step_size)
    local name = call_config.name or config.name
    local base_name = call_config.base_name or config.base_name
    if type(variables) ~= "table" or #variables == 0 then
      error("Lookahead expects a non-empty variable list", 3)
    end

    local options_for_base = merge_tables(base_options, call_base_options,
      call_config.base_options)
    if options_for_base.name == nil then options_for_base.name = base_name end
    local base_targets, base_gradients, base_state =
      base_optimizer(graph, loss, variables, options_for_base)
    if base_targets == nil then
      error("base optimizer returned no update targets", 3)
    end
    if type(base_targets) ~= "table" or base_targets.oper ~= nil then
      base_targets = { base_targets }
    end
    if base_gradients == nil then base_gradients = {} end

    local initializers, state_variables, slow_variables = {}, {}, {}
    local weight_targets = {}
    if type(base_state) == "table" then
      append(initializers, base_state.initializers)
      append(state_variables, base_state.variables)
    end

    local step = Train.variable(graph, 0, name .. "/iterations",
      { dtype = "int32" })
    initializers[#initializers + 1] = step.initializer
    state_variables[#state_variables + 1] = step
    local one = tf.constant(graph, 1, name .. "/step_one", "int32")
    local zero = tf.constant(graph, 0, name .. "/step_zero", "int32")
    local period = tf.constant(graph, sync_period, name .. "/sync_period",
      "int32")
    local next_step = tf.add(graph, step.read, one, name .. "/next_step",
      "int32")
    local remainder = mod(graph, next_step, period, name .. "/step_mod",
      "int32")
    local is_sync = equal(graph, remainder, zero, name .. "/is_sync", "int32")
    local dtype = variables[1].dtype or "float32"

    for index, variable in ipairs(variables) do
      local ref = variable.ref or variable
      local var_dtype = variable.dtype or dtype
      local shape = variable.shape or { }
      local slow = Train.variable(graph, zeros(shape), name .. "/slow_" .. index,
        { dtype = var_dtype })
      local slow_initializer = slow.initializer
      local init_controls = { slow_initializer }
      if variable.initializer ~= nil then
        init_controls[#init_controls + 1] = variable.initializer
      end
      local initial_fast = read_value(graph, ref,
        name .. "/slow_initial_value_" .. index, var_dtype, init_controls)
      local slow_init = assign(graph, slow.ref, initial_fast,
        name .. "/slow_initializer_" .. index, var_dtype)
      slow.initializer = slow_init
      slow_variables[index] = slow
      initializers[#initializers + 1] = slow_init
      state_variables[#state_variables + 1] = slow

      -- Snapshot the fast value after every base update. A value copy is
      -- required here because a plain Identity of a ref can be observed at
      -- different times by the fast and slow branches.
      local fast_after = read_value(graph, ref,
        name .. "/fast_after_" .. index, var_dtype, base_targets)
      local slow_before = read_value(graph, slow.ref,
        name .. "/slow_before_" .. index, var_dtype)
      local difference = tf.sub(graph, fast_after, slow_before,
        name .. "/difference_" .. index, var_dtype)
      local alpha = tf.constant(graph, slow_step_size,
        name .. "/slow_step_size_" .. index, var_dtype)
      local scaled = tf.mul(graph, difference, alpha,
        name .. "/slow_delta_" .. index, var_dtype)
      local blend = tf.add(graph, slow_before, scaled,
        name .. "/blend_" .. index, var_dtype)
      local mask = tf.cast(graph, is_sync, "bool", var_dtype,
        name .. "/sync_mask_" .. index)
      local fast_delta = tf.sub(graph, blend, fast_after,
        name .. "/fast_delta_" .. index, var_dtype)
      local fast_new = tf.add(graph, fast_after,
        tf.mul(graph, mask, fast_delta, name .. "/masked_fast_delta_" .. index,
          var_dtype), name .. "/fast_new_" .. index, var_dtype)
      local slow_new = tf.add(graph, slow_before,
        tf.mul(graph, mask, scaled, name .. "/masked_slow_delta_" .. index,
          var_dtype), name .. "/slow_new_" .. index, var_dtype)
      local fast_assign = assign(graph, ref, fast_new,
        name .. "/fast_assign_" .. index, var_dtype, base_targets)
      local slow_assign = assign(graph, slow.ref, slow_new,
        name .. "/slow_assign_" .. index, var_dtype, base_targets)
      weight_targets[#weight_targets + 1] = fast_assign
      weight_targets[#weight_targets + 1] = slow_assign
    end

    local step_assign = assign(graph, step.ref, next_step,
      name .. "/increment_iterations", "int32", weight_targets)
    local targets = {}
    append(targets, base_targets)
    append(targets, weight_targets)
    targets[#targets + 1] = step_assign

    return targets, base_gradients, {
      initializers = initializers,
      variables = state_variables,
      slow_weights = slow_variables,
      slow_variables = slow_variables,
      slow = slow_variables,
      step = step,
      iterations = step,
      global_step = step,
      sync_period = sync_period,
      slow_step_size = slow_step_size,
      base_optimizer = base_optimizer,
      base_state = base_state,
      optimizer_state = base_state,
    }
  end
end

local function finite_nonnegative(value, label)
  if type(value) ~= "number" or value ~= value or value < 0 or
      value == math.huge then
    error(label .. " must be a finite non-negative number", 4)
  end
  return value
end

-- Conditional gradient (Frank-Wolfe) optimizer with the Frobenius-norm
-- constraint used by tfa.optimizers.ConditionalGradient.  Each variable is
-- updated as lr * variable - (1 - lr) * lambda * gradient / (||gradient|| + eps).
function Optimizers.ConditionalGradient(graph, loss, variables, options)
  options = options or {}
  if type(variables) ~= "table" or #variables == 0 then
    error("ConditionalGradient expects a non-empty variable list", 3)
  end
  local learning_rate = options.learning_rate or options.lr or 0.001
  local lambda = options.lambda_ or options.lambda or 0.01
  local epsilon = options.epsilon or 1e-7
  local ord = options.ord or "fro"
  if ord ~= "fro" then
    error("ConditionalGradient currently supports ord = 'fro'", 3)
  end
  finite_nonnegative(learning_rate, "ConditionalGradient learning_rate")
  finite_nonnegative(lambda, "ConditionalGradient lambda")
  if type(epsilon) ~= "number" or epsilon ~= epsilon or epsilon <= 0 or
      epsilon == math.huge then
    error("ConditionalGradient epsilon must be a finite positive number", 3)
  end
  local name = options.name or "ConditionalGradient"
  local refs = {}
  for index, variable in ipairs(variables) do
    refs[index] = variable.ref or variable
  end
  local gradients = options.gradients or tf.gradients(graph, { loss }, refs)
  local dtype = variables[1].dtype or "float32"
  local lr = tf.constant(graph, learning_rate, name .. "/learning_rate", dtype)
  local lambda_op = tf.constant(graph, lambda, name .. "/lambda", dtype)
  local epsilon_op = tf.constant(graph, epsilon, name .. "/epsilon", dtype)
  local one = tf.constant(graph, 1, name .. "/one", dtype)
  local one_minus_lr = tf.sub(graph, one, lr, name .. "/one_minus_lr", dtype)
  local targets = {}

  for index, variable in ipairs(variables) do
    local ref = variable.ref or variable
    local var_dtype = variable.dtype or dtype
    local gradient = gradients[index]
    if gradient == nil then
      error("ConditionalGradient received no gradient for variable " .. index, 3)
    end
    local squared = tf.square(graph, gradient, name .. "/squared_" .. index,
      var_dtype)
    local norm = tf.sqrt(graph, Ops.sum_all(graph, squared,
      name .. "/norm_squared_" .. index, var_dtype), name .. "/norm_" .. index,
      var_dtype)
    local denominator = tf.add(graph, norm, epsilon_op,
      name .. "/denominator_" .. index, var_dtype)
    local direction = tf.div(graph, gradient, denominator,
      name .. "/direction_" .. index, var_dtype)
    local decayed = tf.mul(graph, variable.read, lr,
      name .. "/decayed_" .. index, var_dtype)
    local step = tf.mul(graph, tf.mul(graph, one_minus_lr, lambda_op,
      name .. "/constraint_" .. index, var_dtype), direction,
      name .. "/step_" .. index, var_dtype)
    local next_weight = tf.sub(graph, decayed, step,
      name .. "/next_weight_" .. index, var_dtype)
    targets[#targets + 1] = assign(graph, ref, next_weight,
      name .. "/weight_assign_" .. index, var_dtype)
  end

  return targets, gradients, {
    initializers = {},
    variables = {},
    learning_rate = lr,
    lambda = lambda_op,
    epsilon = epsilon_op,
    ord = ord,
  }
end

-- Decoupled weight-decay wrapper.  The base optimizer is responsible for
-- applying the decay (the core Adam/SGD implementations support weight_decay).
function Optimizers.WeightDecay(base_optimizer, options)
  if type(base_optimizer) == "table" and options == nil then
    options = base_optimizer
    base_optimizer = options.base_optimizer or options.base or options.optimizer
  end
  base_optimizer = resolve_optimizer(base_optimizer)
  if not is_callable(base_optimizer) then
    error("WeightDecay expects a callable base optimizer", 2)
  end
  local config = options or {}
  local decay = config.weight_decay or config.wd or 0.01
  return function(graph, loss, variables, call_options)
    local merged = merge_tables(config, call_options or {})
    merged.weight_decay = decay
    merged.wd = decay
    return base_optimizer(graph, loss, variables, merged)
  end
end


return Optimizers
