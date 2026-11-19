local tf = require("tensorflow")

local Image = {}

local function reverse(graph, input, axes, name)
  local desc = graph:new_operation("ReverseV2", name)
  desc:add_input(input)
  desc:add_input(tf.constant(graph, axes, name .. "/axes", "int32"))
  desc:set_type("T", input.dtype or "float32")
  desc:set_type("Tidx", "int32")
  return desc:finish()
end

function Image.flip_left_right(graph, input, name)
  return reverse(graph, input, { 2 }, name or "flip_left_right")
end

function Image.flip_up_down(graph, input, name)
  return reverse(graph, input, { 1 }, name or "flip_up_down")
end

function Image.adjust_brightness(graph, input, delta, name)
  name = name or "adjust_brightness"
  local dtype = input.dtype or "float32"
  local value = tf.add(graph, input, tf.constant(graph, delta,
    name .. "/delta", dtype), name, dtype)
  return value
end

function Image.resize_nearest(graph, input, height, width, name)
  name = name or "resize_nearest"
  local dtype = input.dtype or "float32"
  local desc = graph:new_operation("ResizeNearestNeighbor", name)
  desc:add_input(input)
  desc:add_input(tf.constant(graph, { height, width }, name .. "/size",
    "int32"))
  desc:set_type("T", dtype)
  desc:set_string("align_corners", "false")
  return desc:finish()
end

return Image
