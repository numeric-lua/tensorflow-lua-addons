local tf = require("tensorflow")

local Activations = {}

local function coefficients(graph, name, dtype)
  return {
    half = tf.constant(graph, 0.5, name .. "/half", dtype),
    one = tf.constant(graph, 1, name .. "/one", dtype),
    scale = tf.constant(graph, 0.7978845608028654, name .. "/scale", dtype),
    cubic = tf.constant(graph, 0.044715, name .. "/cubic", dtype),
  }
end

function Activations.gelu(graph, input, name, dtype)
  name = name or "gelu"
  dtype = dtype or "float32"
  local c = coefficients(graph, name, dtype)
  local cubed = tf.mul(graph, tf.square(graph, input, name .. "/square", dtype),
    input, name .. "/cubed", dtype)
  local inner = tf.mul(graph, c.scale, tf.add(graph, input,
    tf.mul(graph, c.cubic, cubed, name .. "/cubic_term", dtype),
    name .. "/inner_sum", dtype), name .. "/inner", dtype)
  local tanh = tf.nn.tanh(graph, inner, name .. "/tanh", dtype)
  local gate = tf.add(graph, c.one, tanh, name .. "/gate", dtype)
  return tf.mul(graph, tf.mul(graph, c.half, input, name .. "/half_input",
    dtype), gate, name .. "/output", dtype)
end

function Activations.swish(graph, input, name, dtype)
  name = name or "swish"
  dtype = dtype or "float32"
  return tf.mul(graph, input, tf.nn.sigmoid(graph, input, name .. "/sigmoid",
    dtype), name .. "/output", dtype)
end

function Activations.mish(graph, input, name, dtype)
  name = name or "mish"
  dtype = dtype or "float32"
  local softplus = tf.nn.softplus(graph, input, name .. "/softplus", dtype)
  return tf.mul(graph, input, tf.nn.tanh(graph, softplus,
    name .. "/tanh", dtype), name .. "/output", dtype)
end

function Activations.lisht(graph, input, name, dtype)
  name = name or "lisht"
  dtype = dtype or "float32"
  return tf.mul(graph, input, tf.nn.tanh(graph, input, name .. "/tanh", dtype),
    name .. "/output", dtype)
end

function Activations.tanhshrink(graph, input, name, dtype)
  name = name or "tanhshrink"
  dtype = dtype or "float32"
  return tf.sub(graph, input, tf.nn.tanh(graph, input, name .. "/tanh", dtype),
    name .. "/output", dtype)
end

local function cos_op(graph, input, name, dtype)
  local desc = graph:new_operation("Cos", name)
  desc:add_input(input)
  desc:set_type("T", dtype)
  return desc:finish()
end

-- Snake activation: x + (1 - cos(2 * frequency * x)) / (2 * frequency).
-- Frequency may be a Lua number or a scalar graph tensor.
function Activations.snake(graph, input, frequency, name, dtype)
  name = name or "snake"
  dtype = dtype or "float32"
  frequency = frequency or 1
  local frequency_op
  if type(frequency) == "number" then
    if frequency ~= frequency or frequency <= 0 or frequency == math.huge then
      error("snake frequency must be a finite positive number", 3)
    end
    frequency_op = tf.constant(graph, frequency, name .. "/frequency", dtype)
  else
    frequency_op = tf.cast(graph, frequency, frequency.dtype or dtype, dtype,
      name .. "/frequency")
  end
  local two = tf.constant(graph, 2, name .. "/two", dtype)
  local twice_frequency = tf.mul(graph, two, frequency_op,
    name .. "/twice_frequency", dtype)
  local angle = tf.mul(graph, twice_frequency, input, name .. "/angle", dtype)
  local cosine = cos_op(graph, angle, name .. "/cos", dtype)
  local one = tf.constant(graph, 1, name .. "/one", dtype)
  local numerator = tf.sub(graph, one, cosine, name .. "/numerator", dtype)
  local denominator = tf.mul(graph, two, frequency_op,
    name .. "/denominator", dtype)
  local periodic = tf.div(graph, numerator, denominator,
    name .. "/periodic", dtype)
  return tf.add(graph, input, periodic, name .. "/output", dtype)
end

-- Hard sigmoid: clip(0.2 * x + 0.5, 0, 1).
function Activations.hard_sigmoid(graph, input, name, dtype)
  name = name or "hard_sigmoid"
  dtype = dtype or "float32"
  local scaled = tf.add(graph, tf.mul(graph,
    tf.constant(graph, 0.2, name .. "/slope", dtype), input,
    name .. "/scaled", dtype), tf.constant(graph, 0.5, name .. "/offset",
    dtype), name .. "/shifted", dtype)
  return tf.clip_by_value(graph, scaled, 0, 1, name .. "/output", dtype)
end

function Activations.softsign(graph, input, name, dtype)
  name = name or "softsign"
  dtype = dtype or "float32"
  local denominator = tf.add(graph, tf.constant(graph, 1, name .. "/one",
    dtype), tf.abs(graph, input, name .. "/abs", dtype),
    name .. "/denominator", dtype)
  return tf.div(graph, input, denominator, name .. "/output", dtype)
end

function Activations.softplus(graph, input, name, dtype)
  name = name or "softplus"
  dtype = dtype or "float32"
  return tf.nn.softplus(graph, input, name .. "/output", dtype)
end

local function elu_value(graph, input, alpha, name, dtype)
  local positive = tf.greater(graph, input, tf.constant(graph, 0,
    name .. "/zero", dtype), name .. "/positive", dtype)
  local negative = tf.mul(graph, tf.constant(graph, alpha, name .. "/alpha",
    dtype), tf.sub(graph, tf.exp(graph, input, name .. "/exp", dtype),
    tf.constant(graph, 1, name .. "/one", dtype), name .. "/exp_minus_one",
    dtype), name .. "/negative", dtype)
  return tf.where(graph, positive, input, negative, name .. "/output", dtype)
end

-- Exponential linear unit: x for positive inputs, alpha*(exp(x)-1) otherwise.
function Activations.elu(graph, input, alpha, name, dtype)
  if type(alpha) == "string" then
    dtype = name
    name = alpha
    alpha = nil
  end
  name = name or "elu"
  dtype = dtype or "float32"
  alpha = alpha or 1
  if type(alpha) ~= "number" or alpha ~= alpha or alpha == math.huge then
    error("elu alpha must be a finite number", 3)
  end
  return elu_value(graph, input, alpha, name, dtype)
end

-- Scaled exponential linear unit with the standard SELU constants.
function Activations.selu(graph, input, name, dtype)
  name = name or "selu"
  dtype = dtype or "float32"
  local alpha = 1.6732632423543772
  local scale = 1.0507009873554805
  local value = elu_value(graph, input, alpha, name .. "/elu", dtype)
  return tf.mul(graph, value, tf.constant(graph, scale, name .. "/scale",
    dtype), name .. "/output", dtype)
end

function Activations.relu6(graph, input, name, dtype)
  name = name or "relu6"
  dtype = dtype or "float32"
  return tf.clip_by_value(graph, input, 0, 6, name .. "/output", dtype)
end

-- Leaky ReLU: x for positive inputs, alpha*x otherwise.
function Activations.leaky_relu(graph, input, alpha, name, dtype)
  if type(alpha) == "string" then
    dtype = name
    name = alpha
    alpha = nil
  end
  name = name or "leaky_relu"
  dtype = dtype or "float32"
  alpha = alpha or 0.2
  if type(alpha) ~= "number" or alpha ~= alpha or alpha == math.huge or
      alpha == -math.huge then
    error("leaky_relu alpha must be a finite number", 3)
  end
  local positive = tf.greater(graph, input, tf.constant(graph, 0,
    name .. "/zero", dtype), name .. "/positive", dtype)
  local negative = tf.mul(graph, input, tf.constant(graph, alpha,
    name .. "/alpha", dtype), name .. "/negative", dtype)
  return tf.where(graph, positive, input, negative, name .. "/output", dtype)
end

function Activations.exponential(graph, input, name, dtype)
  name = name or "exponential"
  dtype = dtype or "float32"
  return tf.exp(graph, input, name .. "/output", dtype)
end

function Activations.softmax(graph, input, name, dtype)
  name = name or "softmax"
  dtype = dtype or "float32"
  return tf.nn.softmax(graph, input, name .. "/output", dtype)
end

function Activations.log_softmax(graph, input, name, dtype)
  name = name or "log_softmax"
  dtype = dtype or "float32"
  return tf.nn.log_softmax(graph, input, name .. "/output", dtype)
end

return Activations
