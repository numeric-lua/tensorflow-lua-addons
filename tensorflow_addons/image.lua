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

local function pack_int32(graph, values, name)
  local desc = graph:new_operation("Pack", name)
  desc:add_inputs(values)
  desc:set_type("T", "int32")
  desc:set_int("axis", 0)
  desc:set_int("N", #values)
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

local function depthwise_conv(graph, input, filter, padding, name, dtype)
  local desc = graph:new_operation("DepthwiseConv2dNative", name)
  desc:add_input(input)
  desc:add_input(filter)
  desc:set_type("T", dtype)
  desc:set_int_list("strides", { 1, 1, 1, 1 })
  desc:set_int_list("dilations", { 1, 1, 1, 1 })
  desc:set_string("padding", padding)
  desc:set_string("data_format", "NHWC")
  return desc:finish()
end

local function validate_filter_shape(value, name, allow_even)
  value = value or { 3, 3 }
  if type(value) ~= "table" or #value ~= 2 then
    error(name .. " filter_shape must be {height, width}", 4)
  end
  local height, width = value[1], value[2]
  for _, dimension in ipairs({ height, width }) do
    if type(dimension) ~= "number" or dimension % 1 ~= 0 or dimension < 1 then
      error(name .. " filter_shape dimensions must be positive integers", 4)
    end
    if not allow_even and dimension % 2 == 0 then
      error(name .. " filter_shape dimensions must be odd", 4)
    end
  end
  return height, width
end

local function gaussian_kernel_value(sigma, name)
  if type(sigma) ~= "number" or sigma ~= sigma or sigma <= 0 or
      sigma == math.huge then
    error(name .. " sigma must be a finite positive number", 4)
  end
  return sigma
end

local function gaussian_kernel(filter_shape, sigma, name)
  local height, width = validate_filter_shape(filter_shape, name)
  local sigma_y, sigma_x
  if type(sigma) == "table" then
    sigma_y = gaussian_kernel_value(sigma[1] or 1, name)
    sigma_x = gaussian_kernel_value(sigma[2] or sigma[1] or 1, name)
  else
    sigma_y = gaussian_kernel_value(sigma or 1, name)
    sigma_x = sigma_y
  end
  local center_y = (height - 1) / 2
  local center_x = (width - 1) / 2
  local denominator_y = 2 * sigma_y * sigma_y
  local denominator_x = 2 * sigma_x * sigma_x
  local values, total = {}, 0
  for row = 1, height do
    values[row] = {}
    local delta_y = row - 1 - center_y
    for column = 1, width do
      local delta_x = column - 1 - center_x
      local value = math.exp(-(delta_y * delta_y / denominator_y +
        delta_x * delta_x / denominator_x))
      values[row][column] = { { value } }
      total = total + value
    end
  end
  for row = 1, height do
    for column = 1, width do
      values[row][column][1][1] = values[row][column][1][1] / total
    end
  end
  return values
end

local function tile_kernel(graph, kernel, channels, name, dtype)
  local one = tf.constant(graph, 1, name .. "/one", "int32")
  local multiples = pack_int32(graph, { one, one, channels,
    tf.constant(graph, 1, name .. "/one_channel", "int32") },
    name .. "/multiples")
  local desc = graph:new_operation("Tile", name)
  desc:add_input(kernel)
  desc:add_input(multiples)
  desc:set_type("T", dtype)
  desc:set_type("Tmultiples", "int32")
  return desc:finish()
end

local function pad_image(graph, input, paddings, padding, constant_values,
    name, dtype)
  local paddings_op = tf.constant(graph, paddings, name .. "/paddings",
    "int32")
  local desc
  if padding == "CONSTANT" then
    desc = graph:new_operation("PadV2", name)
    desc:add_input(input)
    desc:add_input(paddings_op)
    desc:add_input(tf.constant(graph, constant_values,
      name .. "/constant_value", dtype))
  else
    desc = graph:new_operation("MirrorPad", name)
    desc:add_input(input)
    desc:add_input(paddings_op)
    desc:set_string("mode", padding)
  end
  desc:set_type("T", dtype)
  desc:set_type("Tpaddings", "int32")
  return desc:finish()
end

-- Gaussian blur with tfa.image.gaussian_filter2d-style options.  Inputs are
-- rank-4 NHWC images (or rank-3 HWC when options.rank = 3).
function Image.gaussian_filter2d(graph, input, options)
  options = options or {}
  local name = options.name or "gaussian_filter2d"
  local dtype = options.dtype or input.dtype or "float32"
  if dtype ~= "float32" and dtype ~= "float64" and dtype ~= "float16" then
    error(name .. " requires a floating point input dtype", 3)
  end
  local rank = options.rank or
    (type(options.input_shape) == "table" and #options.input_shape) or 4
  if rank ~= 3 and rank ~= 4 then
    error(name .. " supports rank-3 or rank-4 inputs", 3)
  end
  local padding = string.upper(options.padding or "REFLECT")
  if padding ~= "REFLECT" and padding ~= "SYMMETRIC" and
      padding ~= "CONSTANT" and padding ~= "SAME" and padding ~= "VALID" then
    error(name .. " padding must be REFLECT, SYMMETRIC, CONSTANT, SAME, or VALID",
      3)
  end
  local kernel_values = gaussian_kernel(options.filter_shape, options.sigma,
    name)
  local height, width = validate_filter_shape(options.filter_shape, name)
  local image = input
  if rank == 3 then
    image = tf.expand_dims(graph, input, 0, name .. "/batch", dtype)
  end
  local channels = last_dimension(graph, image, name .. "/channels", dtype)
  local kernel = tf.constant(graph, kernel_values, name .. "/kernel", dtype)
  kernel = tile_kernel(graph, kernel, channels, name .. "/tiled_kernel", dtype)
  local conv_padding = "VALID"
  if padding == "SAME" or padding == "VALID" then
    conv_padding = padding
  else
    local pad_y = math.floor((height - 1) / 2)
    local pad_x = math.floor((width - 1) / 2)
    image = pad_image(graph, image, {
      { 0, 0 }, { pad_y, pad_y }, { pad_x, pad_x }, { 0, 0 },
    }, padding, options.constant_values or 0, name .. "/padded", dtype)
  end
  local output = depthwise_conv(graph, image, kernel, conv_padding,
    name .. "/conv", dtype)
  if rank == 3 then
    output = tf.squeeze(graph, output, 0, name .. "/unbatch", dtype)
  end
  return output
end

local function image_rank(name, options)
  local rank = options.rank or
    (type(options.input_shape) == "table" and #options.input_shape) or 4
  if rank ~= 2 and rank ~= 3 and rank ~= 4 then
    error(name .. " supports rank-2, rank-3, or rank-4 inputs", 4)
  end
  return rank
end

local function to_4d_image(graph, input, rank, name, dtype)
  if rank == 2 then
    local batched = tf.expand_dims(graph, input, 0, name .. "/batch", dtype)
    return tf.expand_dims(graph, batched, 3, name .. "/channel", dtype)
  end
  if rank == 3 then
    return tf.expand_dims(graph, input, 0, name .. "/batch", dtype)
  end
  return input
end

local function from_4d_image(graph, input, rank, name, dtype)
  if rank == 2 then
    return tf.squeeze(graph, input, { 0, 3 }, name .. "/unbatch", dtype)
  end
  if rank == 3 then
    return tf.squeeze(graph, input, 0, name .. "/unbatch", dtype)
  end
  return input
end

local function reshape_dynamic(graph, input, shape, name, dtype)
  local desc = graph:new_operation("Reshape", name)
  desc:add_input(input)
  desc:add_input(shape)
  desc:set_type("T", dtype)
  desc:set_type("Tshape", "int32")
  return desc:finish()
end

local function extract_image_patches(graph, input, height, width, name, dtype)
  local desc = graph:new_operation("ExtractImagePatches", name)
  desc:add_input(input)
  desc:set_type("T", dtype)
  desc:set_int_list("ksizes", { 1, height, width, 1 })
  desc:set_int_list("strides", { 1, 1, 1, 1 })
  desc:set_int_list("rates", { 1, 1, 1, 1 })
  desc:set_string("padding", "VALID")
  return desc:finish()
end

local function compute_dtype(dtype)
  if dtype == "float16" or dtype == "float32" or dtype == "float64" then
    return dtype
  end
  return "float32"
end

-- Median filtering, matching tfa.image.median_filter2d's filter_shape,
-- padding, and constant_values options.  Supports rank-2, rank-3, and rank-4
-- images and both odd and even filter sizes.
function Image.median_filter2d(graph, input, options)
  options = options or {}
  local name = options.name or "median_filter2d"
  local source_dtype = options.dtype or input.dtype or "float32"
  local dtype = compute_dtype(source_dtype)
  local rank = image_rank(name, options)
  local padding = string.upper(options.padding or "REFLECT")
  if padding ~= "REFLECT" and padding ~= "SYMMETRIC" and
      padding ~= "CONSTANT" then
    error(name .. " padding must be REFLECT, SYMMETRIC, or CONSTANT", 3)
  end
  local height, width = validate_filter_shape(options.filter_shape, name, true)
  local area = height * width
  local image = input
  if source_dtype ~= dtype then
    image = tf.cast(graph, input, source_dtype, dtype, name .. "/compute_input")
  end
  image = to_4d_image(graph, image, rank, name .. "/image", dtype)
  local pad_top = math.floor((height - 1) / 2)
  local pad_left = math.floor((width - 1) / 2)
  image = pad_image(graph, image, {
    { 0, 0 },
    { pad_top, height - 1 - pad_top },
    { pad_left, width - 1 - pad_left },
    { 0, 0 },
  }, padding, options.constant_values or 0, name .. "/padded", dtype)
  local patches = extract_image_patches(graph, image, height, width,
    name .. "/patches", dtype)
  local patch_shape = tf.shape(graph, patches, name .. "/patch_shape", dtype)
  local function dimension(index, label)
    return tf.gather(graph, patch_shape, tf.constant(graph, index,
      name .. "/" .. label .. "_index", "int32"), 0,
      name .. "/" .. label, "int32")
  end
  local batch = dimension(0, "batch")
  local rows = dimension(1, "rows")
  local columns = dimension(2, "columns")
  local channels = last_dimension(graph, image, name .. "/channels", dtype)
  local target_shape = pack_int32(graph, {
    batch,
    rows,
    columns,
    tf.constant(graph, area, name .. "/area", "int32"),
    channels,
  }, name .. "/target_shape")
  local grouped = reshape_dynamic(graph, patches, target_shape,
    name .. "/grouped", dtype)
  local transposed = tf.transpose(graph, grouped, { 0, 1, 2, 4, 3 },
    name .. "/channels_last", dtype)
  local top_k = tf.constant(graph, math.floor(area / 2) + 1,
    name .. "/top_k", "int32")
  local desc = graph:new_operation("TopKV2", name .. "/top_values")
  desc:add_input(transposed)
  desc:add_input(top_k)
  desc:set_type("T", dtype)
  desc:set_bool("sorted", true)
  local top = desc:finish()
  local top_values = { oper = top.oper, index = 0,
    name = name .. "/top_values:0", dtype = dtype }
  local floor = math.floor((area + 1) / 2)
  local ceil = math.floor(area / 2) + 1
  local function column(index, label)
    return tf.gather(graph, top_values, tf.constant(graph, index,
      name .. "/" .. label .. "_index", "int32"), 4,
      name .. "/" .. label, dtype)
  end
  local median
  if area % 2 == 1 then
    median = column(floor - 1, "median")
  else
    local lower = column(floor - 1, "lower_median")
    local upper = column(ceil - 1, "upper_median")
    local total = tf.add(graph, lower, upper, name .. "/median_sum", dtype)
    local two = tf.constant(graph, 2, name .. "/two", dtype)
    median = tf.div(graph, total, two, name .. "/median", dtype)
  end
  if source_dtype ~= dtype then
    median = tf.cast(graph, median, dtype, source_dtype, name .. "/output")
  end
  return from_4d_image(graph, median, rank, name, source_dtype)
end

return Image