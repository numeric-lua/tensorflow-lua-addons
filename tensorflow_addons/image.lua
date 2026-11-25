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

-- Adjust color saturation, following tf.image.adjust_saturation.  The op works
-- per RGB pixel; saturation 0 produces grayscale and 1 preserves the input.
function Image.adjust_saturation(graph, input, saturation, name)
  name = name or "adjust_saturation"
  local dtype = input.dtype or "float32"
  if type(saturation) ~= "number" or saturation ~= saturation or
      saturation < 0 or saturation == math.huge then
    error("adjust_saturation requires a finite non-negative saturation", 3)
  end
  local desc = graph:new_operation("AdjustSaturation", name)
  desc:add_input(input)
  desc:add_input(tf.constant(graph, saturation, name .. "/scale", "float32"))
  desc:set_type("T", dtype)
  return desc:finish()
end

Image.AdjustSaturation = Image.adjust_saturation

-- Rotate hue, following tf.image.adjust_hue.  The delta is expressed in the
-- same normalized hue units as TensorFlow's helper (usually [-1, 1]).
function Image.adjust_hue(graph, input, delta, name)
  name = name or "adjust_hue"
  local dtype = input.dtype or "float32"
  if type(delta) ~= "number" or delta ~= delta or delta == math.huge or
      delta == -math.huge then
    error("adjust_hue requires a finite hue delta", 3)
  end
  local desc = graph:new_operation("AdjustHue", name)
  desc:add_input(input)
  desc:add_input(tf.constant(graph, delta, name .. "/delta", "float32"))
  desc:set_type("T", dtype)
  return desc:finish()
end

Image.AdjustHue = Image.adjust_hue

local function validate_resize_dimensions(height, width, name)
  for _, value in ipairs({ height, width }) do
    if type(value) ~= "number" or value % 1 ~= 0 or value < 1 then
      error(name .. " height and width must be positive integers", 3)
    end
  end
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

-- Bilinear resize for rank-4 NHWC images.  Half-pixel centers match
-- tf.image.resize(..., method="bilinear") by default.
function Image.bilinear_resize(graph, input, height, width, name, options)
  options = options or {}
  name = name or "bilinear_resize"
  local dtype = options.dtype or input.dtype or "float32"
  validate_resize_dimensions(height, width, name)
  local desc = graph:new_operation("ResizeBilinear", name)
  desc:add_input(input)
  desc:add_input(tf.constant(graph, { height, width }, name .. "/size",
    "int32"))
  desc:set_type("T", dtype)
  desc:set_bool("align_corners", options.align_corners == true)
  desc:set_bool("half_pixel_centers", options.half_pixel_centers ~= false)
  return desc:finish()
end

Image.BilinearResize = Image.bilinear_resize

-- Area resize for rank-4 NHWC float images.  Area interpolation averages the
-- source pixels covered by each destination pixel, matching
-- tf.image.resize(..., method="area").
function Image.area_resize(graph, input, height, width, name, options)
  options = options or {}
  name = name or "area_resize"
  local dtype = options.dtype or input.dtype or "float32"
  if dtype ~= "float32" and dtype ~= "float64" then
    error(name .. " requires a float32 or float64 input", 3)
  end
  validate_resize_dimensions(height, width, name)
  local desc = graph:new_operation("ResizeArea", name)
  desc:add_input(input)
  desc:add_input(tf.constant(graph, { height, width }, name .. "/size",
    "int32"))
  desc:set_type("T", dtype)
  desc:set_bool("align_corners", options.align_corners == true)
  return desc:finish()
end

Image.AreaResize = Image.area_resize

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

local function apply_projective_transform(graph, image, transforms,
    interpolation, fill_mode, fill_value, output_shape, name, dtype)
  if output_shape == nil then
    local shape = tf.shape(graph, image, name .. "/transform_shape", dtype)
    output_shape = tf.slice(graph, shape, { 1 }, { 2 },
      name .. "/output_shape", "int32")
  else
    output_shape = tf.constant(graph, output_shape, name .. "/output_shape",
      "int32")
  end
  local desc = graph:new_operation("ImageProjectiveTransformV3",
    name .. "/transform")
  desc:add_input(image)
  desc:add_input(transforms)
  desc:add_input(output_shape)
  desc:add_input(tf.constant(graph, fill_value, name .. "/fill_value",
    "float32"))
  desc:set_type("dtype", dtype)
  desc:set_string("interpolation", interpolation)
  desc:set_string("fill_mode", fill_mode)
  return desc:finish()
end

local function random_uniform_scalar(graph, minimum, maximum, name, dtype,
    seed, seed2)
  local desc = graph:new_operation("RandomUniform", name .. "/unit")
  desc:add_input(tf.constant(graph, {}, name .. "/shape", "int32"))
  desc:set_type("dtype", dtype)
  desc:set_int("seed", seed or 0)
  desc:set_int("seed2", seed2 or 0)
  local unit = desc:finish()
  local scaled = tf.mul(graph, unit, tf.constant(graph, maximum - minimum,
    name .. "/range", dtype), name .. "/scaled", dtype)
  return tf.add(graph, scaled, tf.constant(graph, minimum,
    name .. "/minimum", dtype), name, dtype)
end

local function trig(graph, operation, input, name, dtype)
  local desc = graph:new_operation(operation, name)
  desc:add_input(input)
  desc:set_type("T", dtype)
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

-- Mean filtering, matching tfa.image.mean_filter2d's filter_shape, padding, and
-- constant_values options.  Supports rank-2, rank-3, and rank-4 images.
function Image.mean_filter2d(graph, input, options)
  options = options or {}
  local name = options.name or "mean_filter2d"
  local source_dtype = options.dtype or input.dtype or "float32"
  local dtype = compute_dtype(source_dtype)
  local rank = image_rank(name, options)
  local padding = string.upper(options.padding or "REFLECT")
  if padding ~= "REFLECT" and padding ~= "SYMMETRIC" and
      padding ~= "CONSTANT" then
    error(name .. " padding must be REFLECT, SYMMETRIC, or CONSTANT", 3)
  end
  local height, width = validate_filter_shape(options.filter_shape, name, true)
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
  local area = height * width
  local kernel_values = {}
  for row = 1, height do
    kernel_values[row] = {}
    for column = 1, width do
      kernel_values[row][column] = { { 1 / area } }
    end
  end
  local channels = last_dimension(graph, image, name .. "/channels", dtype)
  local kernel = tf.constant(graph, kernel_values, name .. "/kernel", dtype)
  kernel = tile_kernel(graph, kernel, channels, name .. "/tiled_kernel", dtype)
  local output = depthwise_conv(graph, image, kernel, "VALID",
    name .. "/conv", dtype)
  if source_dtype ~= dtype then
    output = tf.cast(graph, output, dtype, source_dtype, name .. "/output")
  end
  return from_4d_image(graph, output, rank, name, source_dtype)
end

-- Sobel edge maps, matching tf.image.sobel_edges.  Inputs are rank-4 NHWC
-- floating point images.  The result has shape [batch, height, width,
-- channels, 2], where the final axis is [dy, dx] for each channel.
function Image.sobel_edges(graph, input, options)
  options = options or {}
  local name = options.name or "sobel_edges"
  local dtype = options.dtype or input.dtype or "float32"
  if dtype ~= "float32" and dtype ~= "float64" then
    error(name .. " requires a float32 or float64 input", 3)
  end
  if options.input_shape ~= nil then
    if type(options.input_shape) ~= "table" or #options.input_shape ~= 4 then
      error(name .. " requires a rank-4 NHWC input", 3)
    end
  end
  local image = input
  if input.dtype ~= nil and input.dtype ~= dtype then
    image = tf.cast(graph, input, input.dtype, dtype, name .. "/input")
  end
  local dx = {
    { -1, 0, 1 },
    { -2, 0, 2 },
    { -1, 0, 1 },
  }
  local dy = {
    { -1, -2, -1 },
    { 0, 0, 0 },
    { 1, 2, 1 },
  }
  local kernel_values = {}
  for row = 1, 3 do
    kernel_values[row] = {}
    for column = 1, 3 do
      kernel_values[row][column] = {
        { dy[row][column], dx[row][column] },
      }
    end
  end
  local padded = pad_image(graph, image, {
    { 0, 0 }, { 1, 1 }, { 1, 1 }, { 0, 0 },
  }, "REFLECT", 0, name .. "/padded", dtype)
  local channels = last_dimension(graph, image, name .. "/channels", dtype)
  local kernel = tf.constant(graph, kernel_values, name .. "/kernel", dtype)
  kernel = tile_kernel(graph, kernel, channels, name .. "/tiled_kernel", dtype)
  local output = depthwise_conv(graph, padded, kernel, "VALID",
    name .. "/conv", dtype)
  local image_shape = tf.shape(graph, image, name .. "/shape", dtype)
  local two = tf.constant(graph, { 2 }, name .. "/two", "int32")
  local shape = tf.concat(graph, { image_shape, two }, 0,
    name .. "/output_shape", "int32")
  return reshape_dynamic(graph, output, shape, name, dtype)
end

local function round_op(graph, input, name, dtype)
  local desc = graph:new_operation("Round", name)
  desc:add_input(input)
  desc:set_type("T", dtype)
  return desc:finish()
end

-- Adjust image sharpness with the PIL-style 3x3 smoothing kernel, following
-- tfa.image.sharpness.  factor = 1 preserves the image and factor = 0 uses
-- the smoothed interior while keeping the original border.
function Image.sharpness(graph, input, factor, name, options)
  options = options or {}
  name = name or "sharpness"
  if type(factor) ~= "number" or factor ~= factor or factor < 0 or
      factor == math.huge then
    error("sharpness factor must be a finite non-negative number", 3)
  end
  local source_dtype = options.dtype or input.dtype or "float32"
  local rank = options.rank or
    (type(options.input_shape) == "table" and #options.input_shape) or
    (type(input.shape) == "table" and #input.shape) or 4
  if rank ~= 3 and rank ~= 4 then
    error(name .. " supports rank-3 or rank-4 inputs", 3)
  end
  local image = to_4d_image(graph, input, rank, name .. "/image",
    source_dtype)
  if source_dtype ~= "float32" then
    image = tf.cast(graph, image, source_dtype, "float32",
      name .. "/compute_input")
  end
  local kernel_values = {}
  for row = 1, 3 do
    kernel_values[row] = {}
    for column = 1, 3 do
      local value = 1 / 13
      if row == 2 and column == 2 then value = 5 / 13 end
      kernel_values[row][column] = { { value } }
    end
  end
  local channels = last_dimension(graph, image, name .. "/channels", "float32")
  local kernel = tf.constant(graph, kernel_values, name .. "/kernel",
    "float32")
  kernel = tile_kernel(graph, kernel, channels, name .. "/tiled_kernel",
    "float32")
  local smoothed = depthwise_conv(graph, image, kernel, "VALID",
    name .. "/conv", "float32")
  local zero = tf.constant(graph, 0, name .. "/zero", "float32")
  local one = tf.constant(graph, 1, name .. "/one", "float32")
  local mask = tf.add(graph, tf.mul(graph, smoothed, zero,
    name .. "/mask_base", "float32"), one, name .. "/mask", "float32")
  local paddings = { { 0, 0 }, { 1, 1 }, { 1, 1 }, { 0, 0 } }
  local padded_mask = tf.pad(graph, mask, paddings, "CONSTANT",
    name .. "/padded_mask", "float32")
  local padded_smoothed = tf.pad(graph, smoothed, paddings, "CONSTANT",
    name .. "/padded_smoothed", "float32")
  local interior = tf.equal(graph, padded_mask, one,
    name .. "/interior", "float32")
  local result = tf.where(graph, interior, padded_smoothed, image,
    name .. "/result", "float32")
  local blended
  if factor == 0 then
    blended = result
  elseif factor == 1 then
    blended = tf.add(graph, image, zero, name .. "/identity", "float32")
  else
    local difference = tf.sub(graph, image, result, name .. "/difference",
      "float32")
    blended = tf.add(graph, result, tf.mul(graph, difference,
      tf.constant(graph, factor, name .. "/factor", "float32"),
      name .. "/scaled_difference", "float32"), name .. "/blended", "float32")
    if factor < 1 then
      blended = round_op(graph, blended, name .. "/rounded", "float32")
    else
      blended = round_op(graph, tf.clip_by_value(graph, blended, 0, 255,
        name .. "/clipped", "float32"), name .. "/rounded", "float32")
    end
  end
  if source_dtype ~= "float32" then
    blended = tf.cast(graph, blended, "float32", source_dtype,
      name .. "/output")
  end
  return from_4d_image(graph, blended, rank, name, source_dtype)
end

Image.Sharpness = Image.sharpness

-- Blend two same-shaped image tensors, following tfa.image.blend.  Factors
-- between 0 and 1 interpolate, factors above 1 extrapolate and clip to 0..255.
function Image.blend(graph, image1, image2, factor, name)
  name = name or "blend"
  if type(factor) ~= "number" or factor ~= factor or factor < 0 or
      factor == math.huge then
    error("blend factor must be a finite non-negative number", 3)
  end
  local dtype = image1.dtype or image2.dtype or "float32"
  local zero = tf.constant(graph, 0, name .. "/zero", dtype)
  if factor == 0 then
    return tf.add(graph, image1, zero, name, dtype)
  end
  if factor == 1 then
    return tf.add(graph, image2, zero, name, dtype)
  end
  local first = tf.cast(graph, image1, image1.dtype or dtype, "float32",
    name .. "/first")
  local second = tf.cast(graph, image2, image2.dtype or dtype, "float32",
    name .. "/second")
  local difference = tf.sub(graph, second, first, name .. "/difference",
    "float32")
  local blended = tf.add(graph, first, tf.mul(graph, difference,
    tf.constant(graph, factor, name .. "/factor", "float32"),
    name .. "/scaled_difference", "float32"), name .. "/blended", "float32")
  if factor < 1 then
    return round_op(graph, blended, name, "float32")
  end
  blended = tf.clip_by_value(graph, blended, 0, 255, name .. "/clipped",
    "float32")
  return round_op(graph, blended, name, "float32")
end

Image.Blend = Image.blend

-- Translate rank-2, rank-3, or rank-4 images with nearest or bilinear
-- interpolation, following tfa.image.translate.  A single translation may be
-- shared by a rank-4 batch, or one [dx, dy] pair may be supplied per image.
function Image.translate(graph, input, translations, options)
  options = options or {}
  local name = options.name or "translate"
  local dtype = options.dtype or input.dtype or "float32"
  local rank = options.rank or
    (type(options.input_shape) == "table" and #options.input_shape) or
    (type(input.shape) == "table" and #input.shape) or 4
  if rank ~= 2 and rank ~= 3 and rank ~= 4 then
    error(name .. " supports rank-2, rank-3, or rank-4 inputs", 3)
  end
  if type(translations) ~= "table" then
    error(name .. " requires a [dx, dy] translation or a list of pairs", 3)
  end
  local pairs
  if type(translations[1]) == "number" then
    pairs = { translations }
  else
    pairs = translations
  end
  if #pairs == 0 then
    error(name .. " requires at least one [dx, dy] translation", 3)
  end
  local rows = {}
  for index, pair in ipairs(pairs) do
    if type(pair) ~= "table" or #pair ~= 2 or
        type(pair[1]) ~= "number" or type(pair[2]) ~= "number" or
        pair[1] ~= pair[1] or pair[2] ~= pair[2] or
        pair[1] == math.huge or pair[1] == -math.huge or
        pair[2] == math.huge or pair[2] == -math.huge then
      error(name .. " translations must be finite [dx, dy] pairs", 3)
    end
    rows[index] = { 1, 0, -pair[1], 0, 1, -pair[2], 0, 0 }
  end
  local fill_value = options.fill_value or 0
  if type(fill_value) ~= "number" or fill_value ~= fill_value or
      fill_value == math.huge or fill_value == -math.huge then
    error(name .. " fill_value must be a finite number", 3)
  end
  local interpolation = string.upper(options.interpolation or "nearest")
  if interpolation ~= "NEAREST" and interpolation ~= "BILINEAR" then
    error(name .. " interpolation must be 'nearest' or 'bilinear'", 3)
  end
  local fill_mode = string.upper(options.fill_mode or "constant")
  if fill_mode ~= "CONSTANT" and fill_mode ~= "REFLECT" and
      fill_mode ~= "WRAP" and fill_mode ~= "NEAREST" then
    error(name .. " fill_mode must be constant, reflect, wrap, or nearest", 3)
  end
  local image = to_4d_image(graph, input, rank, name .. "/image", dtype)
  local transforms = tf.constant(graph, rows, name .. "/transforms", "float32")
  local output = apply_projective_transform(graph, image, transforms,
    interpolation, fill_mode, fill_value, nil, name, dtype)
  return from_4d_image(graph, output, rank, name, dtype)
end

Image.Translate = Image.translate

local function validate_transform_mode(interpolation, fill_mode, name)
  if interpolation ~= "NEAREST" and interpolation ~= "BILINEAR" then
    error(name .. " interpolation must be 'nearest' or 'bilinear'", 3)
  end
  if fill_mode ~= "CONSTANT" and fill_mode ~= "REFLECT" and
      fill_mode ~= "WRAP" and fill_mode ~= "NEAREST" then
    error(name .. " fill_mode must be constant, reflect, wrap, or nearest", 3)
  end
end

-- Apply explicit projective transforms, following tfa.image.transform.  A
-- transform row is [a0, a1, a2, b0, b1, b2, c0, c1] and maps output
-- coordinates back into the input image.
function Image.transform(graph, input, transforms, options)
  options = options or {}
  local name = options.name or "transform"
  local dtype = options.dtype or input.dtype or "float32"
  local rank = options.rank or
    (type(options.input_shape) == "table" and #options.input_shape) or
    (type(input.shape) == "table" and #input.shape) or 4
  if rank ~= 2 and rank ~= 3 and rank ~= 4 then
    error(name .. " supports rank-2, rank-3, or rank-4 inputs", 3)
  end
  if type(transforms) ~= "table" or #transforms == 0 then
    error(name .. " requires a transform vector or a list of transform rows", 3)
  end
  local rows
  if type(transforms[1]) == "number" then
    rows = { transforms }
  else
    rows = transforms
  end
  for _, row in ipairs(rows) do
    if type(row) ~= "table" or #row ~= 8 then
      error(name .. " transform rows must contain eight numbers", 3)
    end
    for _, value in ipairs(row) do
      if type(value) ~= "number" or value ~= value or
          value == math.huge or value == -math.huge then
        error(name .. " transform entries must be finite numbers", 3)
      end
    end
  end
  local interpolation = string.upper(options.interpolation or "nearest")
  local fill_mode = string.upper(options.fill_mode or "constant")
  validate_transform_mode(interpolation, fill_mode, name)
  local fill_value = options.fill_value or 0
  if type(fill_value) ~= "number" or fill_value ~= fill_value or
      fill_value == math.huge or fill_value == -math.huge then
    error(name .. " fill_value must be a finite number", 3)
  end
  local output_shape = options.output_shape
  if output_shape ~= nil then
    if type(output_shape) ~= "table" or #output_shape ~= 2 then
      error(name .. " output_shape must be {height, width}", 3)
    end
    for _, value in ipairs(output_shape) do
      if type(value) ~= "number" or value % 1 ~= 0 or value < 1 then
        error(name .. " output_shape entries must be positive integers", 3)
      end
    end
  end
  local image = to_4d_image(graph, input, rank, name .. "/image", dtype)
  local transform_tensor = tf.constant(graph, rows, name .. "/transforms",
    "float32")
  local output = apply_projective_transform(graph, image, transform_tensor,
    interpolation, fill_mode, fill_value, output_shape, name, dtype)
  return from_4d_image(graph, output, rank, name, dtype)
end

Image.Transform = Image.transform

-- Rotate rank-2, rank-3, or rank-4 images counterclockwise by radians,
-- following tfa.image.rotate.  A scalar angle applies to the whole input; a
-- list supplies one angle per batch image.
function Image.rotate(graph, input, angles, options)
  options = options or {}
  local name = options.name or "rotate"
  local dtype = options.dtype or input.dtype or "float32"
  local rank = options.rank or
    (type(options.input_shape) == "table" and #options.input_shape) or
    (type(input.shape) == "table" and #input.shape) or 4
  if rank ~= 2 and rank ~= 3 and rank ~= 4 then
    error(name .. " supports rank-2, rank-3, or rank-4 inputs", 3)
  end
  local values
  if type(angles) == "number" then
    values = { angles }
  elseif type(angles) == "table" and #angles > 0 then
    values = angles
  else
    error(name .. " requires an angle or a non-empty list of angles", 3)
  end
  for _, angle in ipairs(values) do
    if type(angle) ~= "number" or angle ~= angle or
        angle == math.huge or angle == -math.huge then
      error(name .. " angles must be finite numbers in radians", 3)
    end
  end
  local interpolation = string.upper(options.interpolation or "nearest")
  local fill_mode = string.upper(options.fill_mode or "constant")
  validate_transform_mode(interpolation, fill_mode, name)
  local fill_value = options.fill_value or 0
  if type(fill_value) ~= "number" or fill_value ~= fill_value or
      fill_value == math.huge or fill_value == -math.huge then
    error(name .. " fill_value must be a finite number", 3)
  end
  local image = to_4d_image(graph, input, rank, name .. "/image", dtype)
  local shape = tf.shape(graph, image, name .. "/shape", dtype)
  local function dimension(index, label)
    return tf.cast(graph, tf.gather(graph, shape,
      tf.constant(graph, index, name .. "/" .. label .. "_index", "int32"), 0,
      name .. "/" .. label, "int32"), "int32", "float32",
      name .. "/" .. label .. "_float")
  end
  local height = dimension(1, "height")
  local width = dimension(2, "width")
  local one = tf.constant(graph, 1, name .. "/one", "float32")
  local two = tf.constant(graph, 2, name .. "/two", "float32")
  local zero = tf.constant(graph, 0, name .. "/zero", "float32")
  local negative_one = tf.constant(graph, -1, name .. "/negative_one",
    "float32")
  local width_minus_one = tf.sub(graph, width, one,
    name .. "/width_minus_one", "float32")
  local height_minus_one = tf.sub(graph, height, one,
    name .. "/height_minus_one", "float32")
  local rows = {}
  for index, angle in ipairs(values) do
    local cosine = tf.constant(graph, math.cos(angle),
      name .. "/cos_" .. index, "float32")
    local sine = tf.constant(graph, math.sin(angle),
      name .. "/sin_" .. index, "float32")
    local negative_sine = tf.mul(graph, negative_one, sine,
      name .. "/negative_sine_" .. index, "float32")
    local x_offset = tf.div(graph, tf.sub(graph, width_minus_one,
      tf.sub(graph, tf.mul(graph, cosine, width_minus_one,
        name .. "/cos_width_" .. index, "float32"), tf.mul(graph, sine,
        height_minus_one, name .. "/sin_height_" .. index, "float32"),
        name .. "/rotated_width_" .. index, "float32"),
      name .. "/width_delta_" .. index, "float32"), two,
      name .. "/x_offset_" .. index, "float32")
    local y_offset = tf.div(graph, tf.sub(graph, height_minus_one,
      tf.add(graph, tf.mul(graph, sine, width_minus_one,
        name .. "/sin_width_" .. index, "float32"), tf.mul(graph, cosine,
        height_minus_one, name .. "/cos_height_" .. index, "float32"),
        name .. "/rotated_height_" .. index, "float32"),
      name .. "/height_delta_" .. index, "float32"), two,
      name .. "/y_offset_" .. index, "float32")
    rows[index] = tf.stack(graph, {
      cosine, negative_sine, x_offset, sine, cosine, y_offset, zero, zero,
    }, 0, name .. "/row_" .. index, "float32")
  end
  local transforms
  if #rows == 1 then
    transforms = tf.reshape(graph, rows[1], { 1, 8 },
      name .. "/transforms", "float32")
  else
    transforms = tf.stack(graph, rows, 0, name .. "/transforms", "float32")
  end
  local output = apply_projective_transform(graph, image, transforms,
    interpolation, fill_mode, fill_value, nil, name, dtype)
  return from_4d_image(graph, output, rank, name, dtype)
end

Image.Rotate = Image.rotate

-- Random HSV adjustment in YIQ space, following tfa.image.random_hsv_in_yiq.
-- The image is adjusted with a hue delta in [-max_delta_hue, max_delta_hue]
-- radians and saturation/value scales drawn from the supplied ranges.
function Image.random_hsv_in_yiq(graph, input, options)
  options = options or {}
  local name = options.name or "random_hsv_in_yiq"
  local dtype = options.dtype or input.dtype or "float32"
  if dtype ~= "float32" and dtype ~= "float64" then
    error(name .. " requires a float32 or float64 input", 3)
  end
  local rank = options.rank or
    (type(options.input_shape) == "table" and #options.input_shape) or
    (type(input.shape) == "table" and #input.shape) or 4
  if rank ~= 3 and rank ~= 4 then
    error(name .. " supports rank-3 or rank-4 RGB inputs", 3)
  end
  if type(options.input_shape) == "table" and
      options.input_shape[#options.input_shape] ~= 3 then
    error(name .. " requires a last dimension of 3", 3)
  end
  local max_delta_hue = options.max_delta_hue or 0
  local lower_saturation = options.lower_saturation or 1
  local upper_saturation = options.upper_saturation or 1
  local lower_value = options.lower_value or 1
  local upper_value = options.upper_value or 1
  local function nonnegative(value, label)
    if type(value) ~= "number" or value ~= value or value < 0 or
        value == math.huge then
      error(name .. " " .. label .. " must be a finite non-negative number", 3)
    end
    return value
  end
  nonnegative(max_delta_hue, "max_delta_hue")
  nonnegative(lower_saturation, "lower_saturation")
  nonnegative(upper_saturation, "upper_saturation")
  nonnegative(lower_value, "lower_value")
  nonnegative(upper_value, "upper_value")
  if lower_saturation > upper_saturation then
    error(name .. " lower_saturation must not exceed upper_saturation", 3)
  end
  if lower_value > upper_value then
    error(name .. " lower_value must not exceed upper_value", 3)
  end
  local zero = tf.constant(graph, 0, name .. "/zero", dtype)
  local negative_one = tf.constant(graph, -1, name .. "/negative_one", dtype)
  local delta_hue
  if max_delta_hue == 0 then
    delta_hue = zero
  else
    delta_hue = random_uniform_scalar(graph, -max_delta_hue, max_delta_hue,
      name .. "/delta_hue", dtype, options.seed, options.seed2)
  end
  local saturation
  if lower_saturation == upper_saturation then
    saturation = tf.constant(graph, lower_saturation,
      name .. "/saturation", dtype)
  else
    saturation = random_uniform_scalar(graph, lower_saturation,
      upper_saturation, name .. "/saturation", dtype, options.seed,
      options.seed2)
  end
  local value
  if lower_value == upper_value then
    value = tf.constant(graph, lower_value, name .. "/value", dtype)
  else
    value = random_uniform_scalar(graph, lower_value, upper_value,
      name .. "/value", dtype, options.seed, options.seed2)
  end
  local cosine = trig(graph, "Cos", delta_hue, name .. "/cos", dtype)
  local sine = trig(graph, "Sin", delta_hue, name .. "/sin", dtype)
  local scaled = tf.mul(graph, value, saturation, name .. "/scaled", dtype)
  local vsu = tf.mul(graph, scaled, cosine, name .. "/vsu", dtype)
  local vsw = tf.mul(graph, scaled, sine, name .. "/vsw", dtype)
  local negative_vsw = tf.mul(graph, negative_one, vsw,
    name .. "/negative_vsw", dtype)
  local row_one = tf.stack(graph, { value, zero, zero }, 0,
    name .. "/row_one", dtype)
  local row_two = tf.stack(graph, { zero, vsu, vsw }, 0,
    name .. "/row_two", dtype)
  local row_three = tf.stack(graph, { zero, negative_vsw, vsu }, 0,
    name .. "/row_three", dtype)
  local hsv = tf.stack(graph, { row_one, row_two, row_three }, 0,
    name .. "/hsv", dtype)
  local yiq = tf.constant(graph, {
    { 0.299, 0.596, 0.211 },
    { 0.587, -0.274, -0.523 },
    { 0.114, -0.322, 0.312 },
  }, name .. "/yiq", dtype)
  local yiq_inverse = tf.constant(graph, {
    { 1, 1, 1 },
    { 0.95617069, -0.2726886, -1.103744 },
    { 0.62143257, -0.64681324, 1.70062309 },
  }, name .. "/yiq_inverse", dtype)
  local transform = tf.matmul(graph, tf.matmul(graph, yiq, hsv,
    name .. "/yiq_hsv", dtype), yiq_inverse, name .. "/transform", dtype)
  local flat = tf.reshape(graph, input, { -1, 3 }, name .. "/flat", dtype)
  local transformed = tf.matmul(graph, flat, transform,
    name .. "/transformed_flat", dtype)
  local output_shape = tf.shape(graph, input, name .. "/input_shape", dtype)
  return reshape_dynamic(graph, transformed, output_shape, name, dtype)
end

Image.RandomHSVInYIQ = Image.random_hsv_in_yiq

local function range_int32(graph, limit, name)
  local desc = graph:new_operation("Range", name)
  desc:add_input(tf.constant(graph, 0, name .. "/start", "int32"))
  desc:add_input(limit)
  desc:add_input(tf.constant(graph, 1, name .. "/delta", "int32"))
  desc:set_type("Tidx", "int32")
  return desc:finish()
end

local function tile_dynamic(graph, input, multiples, name, dtype)
  local values = {}
  for index, value in ipairs(multiples) do
    if type(value) == "number" then
      values[index] = tf.constant(graph, value,
        name .. "/multiple_" .. index, "int32")
    else
      values[index] = value
    end
  end
  local desc = graph:new_operation("Tile", name)
  desc:add_input(input)
  desc:add_input(pack_int32(graph, values, name .. "/multiples"))
  desc:set_type("T", dtype)
  desc:set_type("Tmultiples", "int32")
  return desc:finish()
end

local function random_uniform_vector(graph, shape, name, dtype, seed, seed2)
  local desc = graph:new_operation("RandomUniform", name)
  desc:add_input(shape)
  desc:set_type("dtype", dtype)
  desc:set_int("seed", seed or 0)
  desc:set_int("seed2", seed2 or 0)
  return desc:finish()
end

local function floor_float(graph, input, name, dtype)
  local desc = graph:new_operation("Floor", name)
  desc:add_input(input)
  desc:set_type("T", dtype)
  return desc:finish()
end

local function cutout_mask(graph, images, mask_size, center_y, center_x,
    constant_values, name, dtype)
  local shape = tf.shape(graph, images, name .. "/shape", dtype)
  local function dimension(index, label)
    return tf.gather(graph, shape,
      tf.constant(graph, index, name .. "/" .. label .. "_index", "int32"), 0,
      name .. "/" .. label, "int32")
  end
  local height = dimension(1, "height")
  local width = dimension(2, "width")
  local channels = dimension(3, "channels")
  local half_h = math.floor(mask_size[1] / 2)
  local half_w = math.floor(mask_size[2] / 2)
  local zero_int = tf.constant(graph, 0, name .. "/zero_int", "int32")
  local half_h_op = tf.constant(graph, half_h, name .. "/half_h", "int32")
  local half_w_op = tf.constant(graph, half_w, name .. "/half_w", "int32")
  local lower_y = tf.maximum(graph, tf.sub(graph, center_y, half_h_op,
    name .. "/lower_y_raw", "int32"), zero_int, name .. "/lower_y", "int32")
  local upper_y = tf.minimum(graph, tf.add(graph, center_y, half_h_op,
    name .. "/upper_y_raw", "int32"), height, name .. "/upper_y", "int32")
  local lower_x = tf.maximum(graph, tf.sub(graph, center_x, half_w_op,
    name .. "/lower_x_raw", "int32"), zero_int, name .. "/lower_x", "int32")
  local upper_x = tf.minimum(graph, tf.add(graph, center_x, half_w_op,
    name .. "/upper_x_raw", "int32"), width, name .. "/upper_x", "int32")
  local lower_y_4d = tf.reshape(graph, lower_y, { -1, 1, 1, 1 },
    name .. "/lower_y_4d", "int32")
  local upper_y_4d = tf.reshape(graph, upper_y, { -1, 1, 1, 1 },
    name .. "/upper_y_4d", "int32")
  local lower_x_4d = tf.reshape(graph, lower_x, { -1, 1, 1, 1 },
    name .. "/lower_x_4d", "int32")
  local upper_x_4d = tf.reshape(graph, upper_x, { -1, 1, 1, 1 },
    name .. "/upper_x_4d", "int32")
  local y = tf.reshape(graph, range_int32(graph, height, name .. "/y_range"),
    { 1, -1, 1, 1 }, name .. "/y", "int32")
  local x = tf.reshape(graph, range_int32(graph, width, name .. "/x_range"),
    { 1, 1, -1, 1 }, name .. "/x", "int32")
  local inside_y = tf.logical_and(graph, tf.greater_equal(graph, y, lower_y_4d,
    name .. "/inside_y_start", "int32"), tf.less(graph, y, upper_y_4d,
    name .. "/inside_y_end", "int32"), name .. "/inside_y")
  local inside_x = tf.logical_and(graph, tf.greater_equal(graph, x, lower_x_4d,
    name .. "/inside_x_start", "int32"), tf.less(graph, x, upper_x_4d,
    name .. "/inside_x_end", "int32"), name .. "/inside_x")
  local mask = tf.logical_and(graph, inside_y, inside_x, name .. "/mask")
  mask = tile_dynamic(graph, mask, { 1, 1, 1, channels }, name .. "/mask_tile",
    "bool")
  local zero = tf.constant(graph, 0, name .. "/zero", dtype)
  local fill = tf.add(graph, tf.mul(graph, images, zero, name .. "/fill_base",
    dtype), tf.constant(graph, constant_values, name .. "/fill_value", dtype),
    name .. "/fill", dtype)
  return tf.where(graph, mask, fill, images, name .. "/output", dtype)
end

local function cutout_args(offset)
  if type(offset) == "table" and type(offset[1]) == "number" then
    return { offset }, true
  end
  return offset, false
end

-- Apply a rectangular cutout mask with an explicit center offset per image,
-- following tfa.image.cutout.  Images must be rank-4 NHWC.
function Image.cutout(graph, images, mask_size, offset, options)
  options = options or {}
  local name = options.name or "cutout"
  local dtype = options.dtype or images.dtype or "float32"
  if type(mask_size) ~= "table" or #mask_size ~= 2 then
    error(name .. " requires mask_size = {height, width}", 3)
  end
  for _, value in ipairs(mask_size) do
    if type(value) ~= "number" or value % 1 ~= 0 or value < 2 or
        value % 2 ~= 0 then
      error(name .. " mask dimensions must be positive even integers", 3)
    end
  end
  local rows, single = cutout_args(offset or { 0, 0 })
  if type(rows) ~= "table" or #rows == 0 then
    error(name .. " offset must be a pair or a list of pairs", 3)
  end
  local values = {}
  for index, pair in ipairs(rows) do
    if type(pair) ~= "table" or #pair ~= 2 or
        type(pair[1]) ~= "number" or type(pair[2]) ~= "number" or
        pair[1] % 1 ~= 0 or pair[2] % 1 ~= 0 then
      error(name .. " offsets must be integer [height, width] pairs", 3)
    end
    values[index] = pair
  end
  local constant_values = options.constant_values or 0
  if type(constant_values) ~= "number" or constant_values ~= constant_values
      or constant_values == math.huge or constant_values == -math.huge then
    error(name .. " constant_values must be a finite number", 3)
  end
  if type(images.dtype) == "string" and not images.dtype:match("^float") then
    error(name .. " currently supports floating point images", 3)
  end
  local offset_tensor = tf.constant(graph, values, name .. "/offsets", "int32")
  local center_y = tf.gather(graph, offset_tensor,
    tf.constant(graph, 0, name .. "/center_height_index", "int32"), 1,
    name .. "/center_y", "int32")
  local center_x = tf.gather(graph, offset_tensor,
    tf.constant(graph, 1, name .. "/center_width_index", "int32"), 1,
    name .. "/center_x", "int32")
  if single then
    local shape = tf.shape(graph, images, name .. "/batch_shape", dtype)
    local batch = tf.gather(graph, shape,
      tf.constant(graph, 0, name .. "/batch_index", "int32"), 0,
      name .. "/batch", "int32")
    center_y = tile_dynamic(graph, center_y, { batch }, name .. "/center_y_tile",
      "int32")
    center_x = tile_dynamic(graph, center_x, { batch }, name .. "/center_x_tile",
      "int32")
  end
  return cutout_mask(graph, images, mask_size, center_y, center_x,
    constant_values, name, dtype)
end

Image.Cutout = Image.cutout

local function random_center(graph, image_dimension, mask_dimension, batch,
    name, seed, seed2)
  local half = math.floor(mask_dimension / 2)
  local half_op = tf.constant(graph, half, name .. "/half", "int32")
  local upper = tf.maximum(graph, tf.sub(graph, image_dimension, half_op,
    name .. "/upper_raw", "int32"), tf.add(graph, half_op,
    tf.constant(graph, 1, name .. "/one", "int32"), name .. "/lower_bound",
    "int32"), name .. "/upper", "int32")
  local span = tf.maximum(graph, tf.sub(graph, upper, half_op,
    name .. "/span_raw", "int32"), tf.constant(graph, 1, name .. "/span_one",
    "int32"), name .. "/span", "int32")
  local shape = pack_int32(graph, { batch }, name .. "/batch_shape")
  local unit = random_uniform_vector(graph, shape, name .. "/unit", "float32",
    seed, seed2)
  local scaled = tf.mul(graph, unit, tf.cast(graph, span, "int32", "float32",
    name .. "/span_float"), name .. "/scaled", "float32")
  local offset = tf.cast(graph, floor_float(graph, scaled, name .. "/floor",
    "float32"), "float32", "int32", name .. "/offset")
  return tf.add(graph, offset, half_op, name, "int32")
end

-- Apply a cutout mask at independently sampled per-image centers, following
-- tfa.image.random_cutout.  Images must be rank-4 NHWC.
function Image.random_cutout(graph, images, mask_size, options)
  options = options or {}
  local name = options.name or "random_cutout"
  local dtype = options.dtype or images.dtype or "float32"
  if type(mask_size) ~= "table" or #mask_size ~= 2 then
    error(name .. " requires mask_size = {height, width}", 3)
  end
  for _, value in ipairs(mask_size) do
    if type(value) ~= "number" or value % 1 ~= 0 or value < 2 or
        value % 2 ~= 0 then
      error(name .. " mask dimensions must be positive even integers", 3)
    end
  end
  local constant_values = options.constant_values or 0
  if type(constant_values) ~= "number" or constant_values ~= constant_values
      or constant_values == math.huge or constant_values == -math.huge then
    error(name .. " constant_values must be a finite number", 3)
  end
  local shape = tf.shape(graph, images, name .. "/image_shape", dtype)
  local batch = tf.gather(graph, shape,
    tf.constant(graph, 0, name .. "/batch_dim_index", "int32"), 0,
    name .. "/batch_size", "int32")
  local height = tf.gather(graph, shape,
    tf.constant(graph, 1, name .. "/height_dim_index", "int32"), 0,
    name .. "/image_height", "int32")
  local width = tf.gather(graph, shape,
    tf.constant(graph, 2, name .. "/width_dim_index", "int32"), 0,
    name .. "/image_width", "int32")
  local center_y = random_center(graph, height, mask_size[1], batch,
    name .. "/center_y", options.seed, options.seed2)
  local center_x = random_center(graph, width, mask_size[2], batch,
    name .. "/center_x", options.seed, options.seed2)
  return cutout_mask(graph, images, mask_size, center_y, center_x,
    constant_values, name, dtype)
end

Image.RandomCutout = Image.random_cutout

local function numeric_or_tensor(graph, value, name, dtype)
  if type(value) == "number" then
    return tf.constant(graph, value, name, dtype)
  end
  return tf.cast(graph, value, value.dtype or dtype, dtype, name)
end

-- Adjust hue, saturation, and value in YIQ space, following
-- tfa.image.adjust_hsv_in_yiq.  delta_hue is in radians and the scale factors
-- are finite positive numbers or scalar graph tensors.
function Image.adjust_hsv_in_yiq(graph, input, delta_hue, scale_saturation,
    scale_value, options)
  options = options or {}
  local name = options.name or "adjust_hsv_in_yiq"
  local dtype = options.dtype or input.dtype or "float32"
  if dtype ~= "float32" and dtype ~= "float64" then
    error(name .. " requires a float32 or float64 input", 3)
  end
  local rank = options.rank or
    (type(options.input_shape) == "table" and #options.input_shape) or
    (type(input.shape) == "table" and #input.shape) or 4
  if rank ~= 3 and rank ~= 4 then
    error(name .. " supports rank-3 or rank-4 RGB inputs", 3)
  end
  if type(options.input_shape) == "table" and
      options.input_shape[#options.input_shape] ~= 3 then
    error(name .. " requires a last dimension of 3", 3)
  end
  local delta = numeric_or_tensor(graph, delta_hue,
    name .. "/delta_hue", dtype)
  local saturation = numeric_or_tensor(graph, scale_saturation,
    name .. "/scale_saturation", dtype)
  local value = numeric_or_tensor(graph, scale_value,
    name .. "/scale_value", dtype)
  local cosine = trig(graph, "Cos", delta, name .. "/cos", dtype)
  local sine = trig(graph, "Sin", delta, name .. "/sin", dtype)
  local scaled = tf.mul(graph, value, saturation, name .. "/scaled", dtype)
  local vsu = tf.mul(graph, scaled, cosine, name .. "/vsu", dtype)
  local vsw = tf.mul(graph, scaled, sine, name .. "/vsw", dtype)
  local zero = tf.constant(graph, 0, name .. "/zero", dtype)
  local negative_one = tf.constant(graph, -1, name .. "/negative_one", dtype)
  local row_one = tf.stack(graph, { value, zero, zero }, 0,
    name .. "/row_one", dtype)
  local row_two = tf.stack(graph, { zero, vsu, vsw }, 0,
    name .. "/row_two", dtype)
  local row_three = tf.stack(graph, { zero, tf.mul(graph, negative_one, vsw,
    name .. "/negative_vsw", dtype), vsu }, 0, name .. "/row_three", dtype)
  local hsv = tf.stack(graph, { row_one, row_two, row_three }, 0,
    name .. "/hsv", dtype)
  local yiq = tf.constant(graph, {
    { 0.299, 0.596, 0.211 },
    { 0.587, -0.274, -0.523 },
    { 0.114, -0.322, 0.312 },
  }, name .. "/yiq", dtype)
  local yiq_inverse = tf.constant(graph, {
    { 1, 1, 1 },
    { 0.95617069, -0.2726886, -1.103744 },
    { 0.62143257, -0.64681324, 1.70062309 },
  }, name .. "/yiq_inverse", dtype)
  local transform = tf.matmul(graph, tf.matmul(graph, yiq, hsv,
    name .. "/yiq_hsv", dtype), yiq_inverse, name .. "/transform", dtype)
  local flat = tf.reshape(graph, input, { -1, 3 }, name .. "/flat", dtype)
  local transformed = tf.matmul(graph, flat, transform,
    name .. "/transformed_flat", dtype)
  local output_shape = tf.shape(graph, input, name .. "/input_shape", dtype)
  return reshape_dynamic(graph, transformed, output_shape, name, dtype)
end

Image.AdjustHSVInYIQ = Image.adjust_hsv_in_yiq

local function rows_tensor(graph, transforms, name)
  if type(transforms) ~= "table" or #transforms == 0 then
    error(name .. " requires a non-empty transform list", 3)
  end
  if type(transforms[1]) == "number" then
    if #transforms ~= 8 then
      error(name .. " transform vectors must contain eight numbers", 3)
    end
    return tf.constant(graph, { transforms }, name .. "/rows", "float32")
  end
  for _, row in ipairs(transforms) do
    if type(row) ~= "table" or #row ~= 8 then
      error(name .. " transform rows must contain eight numbers", 3)
    end
  end
  return tf.constant(graph, transforms, name .. "/rows", "float32")
end

local function flat_to_matrices(graph, rows, name)
  local zeros = tf.mul(graph, rows, tf.constant(graph, 0, name .. "/zero",
    "float32"), name .. "/zeros", "float32")
  local ones = tf.add(graph, zeros, tf.constant(graph, 1, name .. "/one",
    "float32"), name .. "/ones", "float32")
  local last_column = tf.slice(graph, ones, { 0, 0 }, { -1, 1 },
    name .. "/last_column", "float32")
  local extended = tf.concat(graph, { rows, last_column }, 1,
    name .. "/extended", "float32")
  return tf.reshape(graph, extended, { -1, 3, 3 }, name .. "/matrices",
    "float32")
end

local function batch_matmul(graph, a, b, name)
  local desc = graph:new_operation("BatchMatMulV2", name)
  desc:add_input(a)
  desc:add_input(b)
  desc:set_type("T", "float32")
  desc:set_bool("adj_x", false)
  desc:set_bool("adj_y", false)
  return desc:finish()
end

-- Compose projective transform rows, following tfa.image.compose_transforms.
-- Each item may be an eight-number vector or a matrix of eight-number rows.
function Image.compose_transforms(graph, transforms, options)
  options = options or {}
  local name = options.name or "compose_transforms"
  if type(transforms) ~= "table" or #transforms == 0 then
    error(name .. " requires a non-empty list of transforms", 3)
  end
  local composed
  for index, value in ipairs(transforms) do
    local rows = rows_tensor(graph, value, name .. "/transform_" .. index)
    local matrices = flat_to_matrices(graph, rows,
      name .. "/matrices_" .. index)
    if composed == nil then
      composed = matrices
    else
      composed = batch_matmul(graph, composed, matrices,
        name .. "/composed_" .. index)
    end
  end
  local flat = tf.reshape(graph, composed, { -1, 9 }, name .. "/flat",
    "float32")
  local linear = tf.slice(graph, flat, { 0, 0 }, { -1, 8 },
    name .. "/linear", "float32")
  local scale = tf.slice(graph, flat, { 0, 8 }, { -1, 1 },
    name .. "/scale", "float32")
  return tf.div(graph, linear, scale, name, "float32")
end

Image.ComposeTransforms = Image.compose_transforms

-- Convert angles in radians to projective transform rows, following
-- tfa.image.angles_to_projective_transforms.
function Image.angles_to_projective_transforms(graph, angles, image_height,
    image_width, options)
  options = options or {}
  local name = options.name or "angles_to_projective_transforms"
  if type(image_height) ~= "number" or image_height <= 0 or
      type(image_width) ~= "number" or image_width <= 0 then
    error(name .. " requires positive image dimensions", 3)
  end
  local values = angles
  if type(angles) == "number" then values = { angles } end
  if type(values) ~= "table" or #values == 0 then
    error(name .. " requires an angle or list of angles", 3)
  end
  local rows = {}
  for index, angle in ipairs(values) do
    if type(angle) ~= "number" or angle ~= angle or angle == math.huge or
        angle == -math.huge then
      error(name .. " angles must be finite numbers", 3)
    end
    local cosine = math.cos(angle)
    local sine = math.sin(angle)
    local x_offset = ((image_width - 1) - (cosine * (image_width - 1) -
      sine * (image_height - 1))) / 2
    local y_offset = ((image_height - 1) - (sine * (image_width - 1) +
      cosine * (image_height - 1))) / 2
    rows[index] = { cosine, -sine, x_offset, sine, cosine, y_offset, 0, 0 }
  end
  return tf.constant(graph, rows, name, "float32")
end

Image.AnglesToProjectiveTransforms = Image.angles_to_projective_transforms

return Image
