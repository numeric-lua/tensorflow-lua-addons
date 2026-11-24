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
    interpolation, fill_mode, fill_value, name, dtype)
  local shape = tf.shape(graph, image, name .. "/transform_shape", dtype)
  local output_shape = tf.slice(graph, shape, { 1 }, { 2 },
    name .. "/output_shape", "int32")
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
    interpolation, fill_mode, fill_value, name, dtype)
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
    interpolation, fill_mode, fill_value, name, dtype)
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

return Image
