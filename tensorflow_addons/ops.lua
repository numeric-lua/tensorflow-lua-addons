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

function Ops.equal(graph, a, b, name, dtype)
  local desc = graph:new_operation("Equal", name)
  desc:add_input(a)
  desc:add_input(b)
  desc:set_type("T", dtype or "float32")
  return desc:finish()
end

function Ops.not_equal(graph, a, b, name, dtype)
  local desc = graph:new_operation("NotEqual", name)
  desc:add_input(a)
  desc:add_input(b)
  desc:set_type("T", dtype or "float32")
  return desc:finish()
end

function Ops.less_equal(graph, a, b, name, dtype)
  local desc = graph:new_operation("LessEqual", name)
  desc:add_input(a)
  desc:add_input(b)
  desc:set_type("T", dtype or "float32")
  return desc:finish()
end

function Ops.logical_not(graph, input, name)
  local desc = graph:new_operation("LogicalNot", name)
  desc:add_input(input)
  return desc:finish()
end

function Ops.logical_and(graph, a, b, name)
  local desc = graph:new_operation("LogicalAnd", name)
  desc:add_input(a)
  desc:add_input(b)
  return desc:finish()
end

function Ops.flatten(graph, input, name, dtype)
  return tf.reshape(graph, input, { -1 }, name, dtype)
end

function Ops.split(graph, input, num, axis, name, dtype)
  if type(num) ~= "number" or num < 1 then
    error("split expects a positive output count", 3)
  end
  local axis_op = tf.constant(graph, axis or 0, name .. "/axis", "int32")
  local desc = graph:new_operation("Split", name)
  desc:add_input(axis_op)
  desc:add_input(input)
  desc:set_type("T", dtype or "float32")
  desc:set_int("num_split", num)
  local first = desc:finish()
  local outputs = {}
  for index = 0, num - 1 do
    outputs[index + 1] = { oper = first.oper, index = index,
      name = name .. ":" .. index, dtype = dtype or "float32" }
  end
  return outputs
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
