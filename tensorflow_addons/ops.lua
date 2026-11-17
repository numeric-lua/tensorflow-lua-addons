local Ops = {}
local tf = require("tensorflow")

function Ops.maximum(graph, a, b, name, dtype)
  local desc = graph:new_operation("Maximum", name)
  desc:add_input(a)
  desc:add_input(b)
  desc:set_type("T", dtype or "float32")
  return desc:finish()
end

function Ops.minimum(graph, a, b, name, dtype)
  local desc = graph:new_operation("Minimum", name)
  desc:add_input(a)
  desc:add_input(b)
  desc:set_type("T", dtype or "float32")
  return desc:finish()
end

function Ops.pow(graph, a, b, name, dtype)
  local desc = graph:new_operation("Pow", name)
  desc:add_input(a)
  desc:add_input(b)
  desc:set_type("T", dtype or "float32")
  return desc:finish()
end

function Ops.greater(graph, a, b, name, dtype)
  local desc = graph:new_operation("Greater", name)
  desc:add_input(a)
  desc:add_input(b)
  desc:set_type("T", dtype or "float32")
  return desc:finish()
end

function Ops.flatten(graph, input, name, dtype)
  return tf.reshape(graph, input, { -1 }, name, dtype)
end

function Ops.sum_all(graph, input, name, dtype)
  return tf.reduce_sum(graph, Ops.flatten(graph, input, name .. "/flat",
    dtype), { 0 }, name, dtype)
end

function Ops.clean_div(graph, numerator, denominator, name, dtype, epsilon)
  epsilon = epsilon or 1e-7
  local safe = Ops.maximum(graph, denominator,
    tf.constant(graph, epsilon, name .. "/epsilon", dtype), name .. "/safe",
    dtype)
  return tf.div(graph, numerator, safe, name, dtype)
end

return Ops
