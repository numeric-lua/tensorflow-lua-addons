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

return Activations
