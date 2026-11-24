local tf = require("tensorflow")
local Ops = require("tensorflow_addons.ops")

local Losses = {}

local function prepare(graph, labels, predictions, name, dtype)
  labels = tf.cast(graph, labels, labels.dtype or "float32", dtype,
    name .. "/labels")
  predictions = tf.clip_by_value(graph, predictions, 1e-7, 1 - 1e-7,
    name .. "/clipped", dtype)
  return labels, predictions
end

local function reduce_elementwise(graph, values, options, name, dtype,
    default_reduction)
  local reduction = options.reduction or default_reduction or "mean"
  if reduction == "none" then return values end
  if reduction == "sum" then
    return Ops.sum_all(graph, values, name .. "/sum", dtype)
  end
  if reduction ~= "mean" and reduction ~= "sum_over_batch_size" then
    error(name .. " reduction must be 'none', 'mean', or 'sum'", 5)
  end
  return tf.reduce_mean(graph, Ops.flatten(graph, values, name .. "/flat",
    dtype), { 0 }, name .. "/mean", dtype)
end

local function pack_int32(graph, values, name)
  local inputs = {}
  for index, value in ipairs(values) do
    if type(value) == "number" then
      inputs[index] = tf.constant(graph, value, name .. "/" .. index, "int32")
    else
      inputs[index] = value
    end
  end
  local desc = graph:new_operation("Pack", name)
  desc:add_inputs(inputs)
  desc:set_type("T", "int32")
  desc:set_int("axis", 0)
  desc:set_int("N", #inputs)
  return desc:finish()
end

local function reshape_dynamic(graph, input, shape, name, dtype)
  local desc = graph:new_operation("Reshape", name)
  desc:add_input(input)
  desc:add_input(pack_int32(graph, shape, name .. "/shape"))
  desc:set_type("T", dtype)
  desc:set_type("Tshape", "int32")
  return desc:finish()
end

local function tile_dynamic(graph, input, multiples, name, dtype)
  local desc = graph:new_operation("Tile", name)
  desc:add_input(input)
  desc:add_input(pack_int32(graph, multiples, name .. "/multiples"))
  desc:set_type("T", dtype)
  desc:set_type("Tmultiples", "int32")
  return desc:finish()
end

local function dimension(graph, input, index, name, dtype)
  local shape = tf.shape(graph, input, name .. "/shape", dtype)
  local position = tf.constant(graph, index, name .. "/index", "int32")
  return tf.gather(graph, shape, position, 0, name, "int32")
end

local function matmul(graph, a, b, transpose_a, transpose_b, name, dtype)
  local desc = graph:new_operation("MatMul", name)
  desc:add_input(a)
  desc:add_input(b)
  desc:set_type("T", dtype)
  desc:set_bool("transpose_a", transpose_a == true)
  desc:set_bool("transpose_b", transpose_b == true)
  return desc:finish()
end

local function transpose(graph, input, perm, name, dtype)
  return tf.transpose(graph, input, perm, name, dtype)
end

local function identity_matrix(graph, count, name, dtype)
  local start = tf.constant(graph, 0, name .. "/start", "int32")
  local delta = tf.constant(graph, 1, name .. "/delta", "int32")
  local desc = graph:new_operation("Range", name .. "/range")
  desc:add_input(start)
  desc:add_input(count)
  desc:add_input(delta)
  desc:set_type("Tidx", "int32")
  local indices = desc:finish()
  local row = tf.reshape(graph, indices, { -1, 1 }, name .. "/row", "int32")
  local column = tf.reshape(graph, indices, { 1, -1 }, name .. "/column",
    "int32")
  local same = Ops.equal(graph, row, column, name .. "/equal", "int32")
  return tf.cast(graph, same, "bool", dtype, name)
end

-- Stable log-softmax cross entropy for a square batch similarity matrix whose
-- rows are already normalized target distributions (the N-pairs formulation).
local function softmax_cross_entropy(graph, logits, targets, name, dtype)
  local maximum = tf.reduce_max(graph, logits, { 1 }, name .. "/maximum",
    dtype, true)
  local shifted = tf.sub(graph, logits, maximum, name .. "/shifted", dtype)
  local exponentials = tf.exp(graph, shifted, name .. "/exponentials", dtype)
  local total = tf.reduce_sum(graph, exponentials, { 1 }, name .. "/total",
    dtype, true)
  local log_total = tf.log(graph, total, name .. "/log_total", dtype)
  local log_probabilities = tf.sub(graph, shifted, log_total,
    name .. "/log_probabilities", dtype)
  local weighted = tf.mul(graph, targets, log_probabilities,
    name .. "/weighted", dtype)
  local row_loss = tf.reduce_sum(graph, weighted, { 1 },
    name .. "/row_loss", dtype, false)
  local negative = tf.constant(graph, -1, name .. "/negative", dtype)
  row_loss = tf.mul(graph, negative, row_loss, name .. "/positive_row_loss",
    dtype)
  return tf.reduce_mean(graph, row_loss, { 0 }, name, dtype)
end

local function npairs_common(graph, labels, adjacency, name, dtype)
  local row_sums = tf.reduce_sum(graph, adjacency, { 1 },
    name .. "/row_sums", dtype, true)
  local targets = tf.div(graph, adjacency, row_sums, name .. "/targets", dtype)
  return softmax_cross_entropy(graph, labels, targets, name, dtype)
end

-- Binary hinge loss, following the TensorFlow/Keras hinge contract.  Labels
-- are expected in {-1, 1}; each element is max(0, margin - label * prediction).
-- The default reduction matches Keras' "sum_over_batch_size": "none", "mean",
-- and "sum" are also accepted.
function Losses.hinge(graph, labels, predictions, options)
  options = options or {}
  local name = options.name or "hinge_loss"
  local dtype = options.dtype or "float32"
  local margin = options.margin
  if margin == nil then margin = 1 end
  if type(margin) ~= "number" or margin ~= margin or margin < 0 or
      margin == math.huge then
    error("hinge margin must be a finite non-negative number", 4)
  end
  labels = tf.cast(graph, labels, labels.dtype or "float32", dtype,
    name .. "/labels")
  predictions = tf.cast(graph, predictions, predictions.dtype or "float32",
    dtype, name .. "/predictions")
  local product = tf.mul(graph, labels, predictions, name .. "/product", dtype)
  local violation = tf.sub(graph,
    tf.constant(graph, margin, name .. "/margin", dtype), product,
    name .. "/violation", dtype)
  local elementwise = Ops.maximum(graph, violation,
    tf.constant(graph, 0, name .. "/zero", dtype), name .. "/loss_values",
    dtype)
  return reduce_elementwise(graph, elementwise, options, name, dtype,
    "sum_over_batch_size")
end

Losses.Hinge = Losses.hinge
Losses.hinge_loss = Losses.hinge

-- Kullback-Leibler divergence, following Keras' kl_divergence loss.  `labels`
-- and `predictions` are probability distributions along the last axis; each
-- example contributes sum(labels * log(labels / predictions)).  Zero labels
-- contribute zero, and predictions are floored by `options.epsilon` (default
-- 1e-7) to keep the result finite.
function Losses.kl_divergence(graph, labels, predictions, options)
  options = options or {}
  local name = options.name or "kl_divergence"
  local dtype = options.dtype or "float32"
  local epsilon = options.epsilon
  if epsilon == nil then epsilon = 1e-7 end
  if type(epsilon) ~= "number" or epsilon ~= epsilon or epsilon <= 0 or
      epsilon == math.huge then
    error("kl_divergence epsilon must be a finite positive number", 4)
  end
  labels = tf.cast(graph, labels, labels.dtype or "float32", dtype,
    name .. "/labels")
  predictions = tf.cast(graph, predictions, predictions.dtype or "float32",
    dtype, name .. "/predictions")
  local zero = tf.constant(graph, 0, name .. "/zero", dtype)
  local one = tf.constant(graph, 1, name .. "/one", dtype)
  local positive = Ops.greater(graph, labels, zero, name .. "/positive", dtype)
  local safe_labels = tf.where(graph, positive, labels, one,
    name .. "/safe_labels", dtype)
  local safe_predictions = Ops.maximum(graph, predictions,
    tf.constant(graph, epsilon, name .. "/epsilon", dtype),
    name .. "/safe_predictions", dtype)
  local ratio = tf.div(graph, safe_labels, safe_predictions,
    name .. "/ratio", dtype)
  local log_ratio = tf.log(graph, ratio, name .. "/log_ratio", dtype)
  local terms = tf.mul(graph, safe_labels, log_ratio, name .. "/terms", dtype)
  local elementwise = tf.where(graph, positive, terms, zero,
    name .. "/loss_values", dtype)
  local per_example = tf.reduce_sum(graph, elementwise, { -1 },
    name .. "/per_example", dtype, false)
  return reduce_elementwise(graph, per_example, options, name, dtype,
    "sum_over_batch_size")
end

Losses.KLDivergence = Losses.kl_divergence
Losses.kl_divergence_loss = Losses.kl_divergence

-- Log-cosh loss, matching tfa.losses.LogCosh.  The stable form
-- |x| + softplus(-2|x|) - log(2) avoids overflow for large residuals.
function Losses.log_cosh(graph, labels, predictions, options)
  options = options or {}
  local name = options.name or "log_cosh_loss"
  local dtype = options.dtype or "float32"
  labels = tf.cast(graph, labels, labels.dtype or "float32", dtype,
    name .. "/labels")
  predictions = tf.cast(graph, predictions, predictions.dtype or "float32",
    dtype, name .. "/predictions")
  local difference = tf.sub(graph, predictions, labels,
    name .. "/difference", dtype)
  local absolute = tf.abs(graph, difference, name .. "/absolute", dtype)
  local scaled = tf.mul(graph, tf.constant(graph, -2,
    name .. "/negative_two", dtype), absolute, name .. "/scaled", dtype)
  local softplus = tf.nn.softplus(graph, scaled, name .. "/softplus", dtype)
  local log_two = tf.constant(graph, math.log(2), name .. "/log_two", dtype)
  local elementwise = tf.sub(graph, tf.add(graph, absolute, softplus,
    name .. "/stable_sum", dtype), log_two, name .. "/elementwise", dtype)
  local reduction = options.reduction or "sum_over_batch_size"
  if reduction == "none" then return elementwise end
  if reduction == "sum" then
    return Ops.sum_all(graph, elementwise, name .. "/sum", dtype)
  end
  if reduction ~= "mean" and reduction ~= "sum_over_batch_size" then
    error("log_cosh reduction must be 'none', 'mean', or 'sum'", 3)
  end
  return tf.reduce_mean(graph, Ops.flatten(graph, elementwise,
    name .. "/flat", dtype), { 0 }, name .. "/mean", dtype)
end

Losses.LogCosh = Losses.log_cosh

function Losses.focal(graph, labels, predictions, options)
  options = options or {}
  local name = options.name or "focal_loss"
  local dtype = options.dtype or "float32"
  local gamma = options.gamma or 2
  local alpha = options.alpha or 0.25
  labels, predictions = prepare(graph, labels, predictions, name, dtype)
  local one = tf.constant(graph, 1, name .. "/one", dtype)
  local alpha_op = tf.constant(graph, alpha, name .. "/alpha", dtype)
  local gamma_op = tf.constant(graph, gamma, name .. "/gamma", dtype)
  local positive = Ops.pow(graph, tf.sub(graph, one, predictions,
    name .. "/positive_base", dtype), gamma_op, name .. "/positive_pow", dtype)
  local negative = Ops.pow(graph, predictions, gamma_op,
    name .. "/negative_pow", dtype)
  local positive_loss = tf.mul(graph, tf.mul(graph, alpha_op, labels,
    name .. "/alpha_labels", dtype), tf.mul(graph, positive,
    tf.log(graph, predictions, name .. "/log_positive", dtype),
    name .. "/positive_term", dtype), name .. "/positive_loss", dtype)
  local negative_labels = tf.sub(graph, one, labels,
    name .. "/negative_labels", dtype)
  local negative_loss = tf.mul(graph, tf.mul(graph,
    tf.constant(graph, 1 - alpha, name .. "/one_minus_alpha", dtype),
    negative_labels, name .. "/negative_weighted", dtype),
    tf.mul(graph, negative, tf.log(graph, tf.sub(graph, one, predictions,
      name .. "/negative_base", dtype), name .. "/log_negative", dtype),
      name .. "/negative_term", dtype), name .. "/negative_loss", dtype)
  local loss = tf.mul(graph, tf.constant(graph, -1, name .. "/negative_one",
    dtype), tf.add(graph, positive_loss, negative_loss,
    name .. "/combined", dtype), name .. "/loss", dtype)
  return tf.reduce_mean(graph, loss, { 0, 1 }, name .. "/mean", dtype)
end

function Losses.dice(graph, labels, predictions, options)
  options = options or {}
  local name = options.name or "dice_loss"
  local dtype = options.dtype or "float32"
  local smooth = options.smooth or 1
  labels, predictions = prepare(graph, labels, predictions, name, dtype)
  local intersection = Ops.sum_all(graph, tf.mul(graph, labels, predictions,
    name .. "/product", dtype), name .. "/intersection", dtype)
  local label_sum = Ops.sum_all(graph, labels, name .. "/label_sum", dtype)
  local prediction_sum = Ops.sum_all(graph, predictions,
    name .. "/prediction_sum", dtype)
  local numerator = tf.add(graph, tf.mul(graph,
    tf.constant(graph, 2, name .. "/two", dtype), intersection,
    name .. "/twice_intersection", dtype),
    tf.constant(graph, smooth, name .. "/smooth", dtype),
    name .. "/numerator", dtype)
  local denominator = tf.add(graph, tf.add(graph, label_sum, prediction_sum,
    name .. "/sum", dtype), tf.constant(graph, smooth, name .. "/smooth_den",
    dtype), name .. "/denominator", dtype)
  return tf.sub(graph, tf.constant(graph, 1, name .. "/one", dtype),
    tf.div(graph, numerator, denominator, name .. "/coefficient", dtype),
    name .. "/loss", dtype)
end

function Losses.tversky(graph, labels, predictions, options)
  options = options or {}
  local name = options.name or "tversky_loss"
  local dtype = options.dtype or "float32"
  local alpha = options.alpha or 0.5
  local beta = options.beta or 0.5
  local smooth = options.smooth or 1
  labels, predictions = prepare(graph, labels, predictions, name, dtype)
  local one = tf.constant(graph, 1, name .. "/one", dtype)
  local true_positive = tf.mul(graph, labels, predictions,
    name .. "/true_positive_tensor", dtype)
  local false_positive = tf.mul(graph, tf.sub(graph, one, labels,
    name .. "/one_minus_labels", dtype), predictions,
    name .. "/false_positive_tensor", dtype)
  local false_negative = tf.mul(graph, labels, tf.sub(graph, one, predictions,
    name .. "/one_minus_predictions", dtype),
    name .. "/false_negative_tensor", dtype)
  local tp = Ops.sum_all(graph, true_positive, name .. "/tp", dtype)
  local fp = Ops.sum_all(graph, false_positive, name .. "/fp", dtype)
  local fn = Ops.sum_all(graph, false_negative, name .. "/fn", dtype)
  local numerator = tf.add(graph, tp, tf.constant(graph, smooth,
    name .. "/smooth", dtype), name .. "/numerator", dtype)
  local denominator = tf.add(graph, numerator, tf.add(graph,
    tf.mul(graph, tf.constant(graph, alpha, name .. "/alpha", dtype), fp,
      name .. "/alpha_fp", dtype),
    tf.mul(graph, tf.constant(graph, beta, name .. "/beta", dtype), fn,
      name .. "/beta_fn", dtype), name .. "/weighted_errors", dtype),
    name .. "/denominator", dtype)
  return tf.sub(graph, tf.constant(graph, 1, name .. "/one_result", dtype),
    tf.div(graph, numerator, denominator, name .. "/coefficient", dtype),
    name .. "/loss", dtype)
end

function Losses.giou(graph, labels, predictions, options)
  options = options or {}
  local name = options.name or "giou_loss"
  local dtype = options.dtype or "float32"
  local epsilon = options.epsilon or options.eps or 1e-7
  labels = tf.cast(graph, labels, labels.dtype or "float32", dtype,
    name .. "/labels")
  predictions = tf.cast(graph, predictions, predictions.dtype or "float32",
    dtype, name .. "/predictions")
  local true_boxes = Ops.split(graph, labels, 4, -1, name .. "/true_boxes",
    dtype)
  local pred_boxes = Ops.split(graph, predictions, 4, -1, name .. "/pred_boxes",
    dtype)
  local zero = tf.constant(graph, 0, name .. "/zero", dtype)
  local function ordered(boxes, prefix)
    local x1 = Ops.minimum(graph, boxes[1], boxes[3], prefix .. "/x1", dtype)
    local y1 = Ops.minimum(graph, boxes[2], boxes[4], prefix .. "/y1", dtype)
    local x2 = Ops.maximum(graph, boxes[1], boxes[3], prefix .. "/x2", dtype)
    local y2 = Ops.maximum(graph, boxes[2], boxes[4], prefix .. "/y2", dtype)
    return x1, y1, x2, y2
  end
  local tx1, ty1, tx2, ty2 = ordered(true_boxes, name .. "/true")
  local px1, py1, px2, py2 = ordered(pred_boxes, name .. "/pred")
  local true_width = tf.sub(graph, tx2, tx1, name .. "/true_width", dtype)
  local true_height = tf.sub(graph, ty2, ty1, name .. "/true_height", dtype)
  local pred_width = tf.sub(graph, px2, px1, name .. "/pred_width", dtype)
  local pred_height = tf.sub(graph, py2, py1, name .. "/pred_height", dtype)
  local true_area = tf.mul(graph, true_width, true_height,
    name .. "/true_area", dtype)
  local pred_area = tf.mul(graph, pred_width, pred_height,
    name .. "/pred_area", dtype)
  local intersect_x1 = Ops.maximum(graph, tx1, px1,
    name .. "/intersect_x1", dtype)
  local intersect_y1 = Ops.maximum(graph, ty1, py1,
    name .. "/intersect_y1", dtype)
  local intersect_x2 = Ops.minimum(graph, tx2, px2,
    name .. "/intersect_x2", dtype)
  local intersect_y2 = Ops.minimum(graph, ty2, py2,
    name .. "/intersect_y2", dtype)
  local intersect_width = Ops.maximum(graph, tf.sub(graph, intersect_x2,
    intersect_x1, name .. "/intersect_width_raw", dtype), zero,
    name .. "/intersect_width", dtype)
  local intersect_height = Ops.maximum(graph, tf.sub(graph, intersect_y2,
    intersect_y1, name .. "/intersect_height_raw", dtype), zero,
    name .. "/intersect_height", dtype)
  local intersection = tf.mul(graph, intersect_width, intersect_height,
    name .. "/intersection", dtype)
  local union = tf.sub(graph, tf.add(graph, true_area, pred_area,
    name .. "/area_sum", dtype), intersection, name .. "/union", dtype)
  local enclose_x1 = Ops.minimum(graph, tx1, px1,
    name .. "/enclose_x1", dtype)
  local enclose_y1 = Ops.minimum(graph, ty1, py1,
    name .. "/enclose_y1", dtype)
  local enclose_x2 = Ops.maximum(graph, tx2, px2,
    name .. "/enclose_x2", dtype)
  local enclose_y2 = Ops.maximum(graph, ty2, py2,
    name .. "/enclose_y2", dtype)
  local enclose_width = tf.sub(graph, enclose_x2, enclose_x1,
    name .. "/enclose_width", dtype)
  local enclose_height = tf.sub(graph, enclose_y2, enclose_y1,
    name .. "/enclose_height", dtype)
  local enclose_area = tf.mul(graph, enclose_width, enclose_height,
    name .. "/enclose_area", dtype)
  local iou = Ops.clean_div(graph, intersection, union, name .. "/iou", dtype,
    epsilon)
  local penalty = Ops.clean_div(graph, tf.sub(graph, enclose_area, union,
    name .. "/penalty_raw", dtype), enclose_area, name .. "/penalty", dtype,
    epsilon)
  local giou = tf.sub(graph, iou, penalty, name .. "/giou", dtype)
  local loss_values = tf.sub(graph, tf.constant(graph, 1, name .. "/one",
    dtype), giou, name .. "/loss_values", dtype)
  return tf.reduce_mean(graph, Ops.flatten(graph, loss_values,
    name .. "/loss_flat", dtype), { 0 }, name .. "/loss", dtype)
end

Losses.giou_loss = Losses.giou

function Losses.contrastive(graph, labels, left, right, options)
  options = options or {}
  local name = options.name or "contrastive_loss"
  local dtype = options.dtype or "float32"
  local margin = options.margin or 1
  labels = tf.cast(graph, labels, labels.dtype or "float32", dtype,
    name .. "/labels")
  local delta = tf.sub(graph, left, right, name .. "/delta", dtype)
  local squared = tf.square(graph, delta, name .. "/squared", dtype)
  local distances = tf.reduce_sum(graph, squared, { 1 }, name .. "/squared_distance",
    dtype)
  local distance = tf.sqrt(graph, distances, name .. "/distance", dtype)
  local half = tf.constant(graph, 0.5, name .. "/half", dtype)
  local one = tf.constant(graph, 1, name .. "/one", dtype)
  local margin_op = tf.constant(graph, margin, name .. "/margin", dtype)
  local positive = tf.mul(graph, half, tf.mul(graph, labels,
    tf.square(graph, distance, name .. "/positive_square", dtype),
    name .. "/positive_weighted", dtype), name .. "/positive", dtype)
  local negative_distance = Ops.maximum(graph, tf.sub(graph, margin_op,
    distance, name .. "/margin_minus_distance", dtype),
    tf.constant(graph, 0, name .. "/zero", dtype), name .. "/negative_distance",
    dtype)
  local negative = tf.mul(graph, half, tf.mul(graph,
    tf.sub(graph, one, labels, name .. "/one_minus_labels", dtype),
    tf.square(graph, negative_distance, name .. "/negative_square", dtype),
    name .. "/negative_weighted", dtype), name .. "/negative", dtype)
  return tf.reduce_mean(graph, tf.add(graph, positive, negative,
    name .. "/loss_values", dtype), { 0 }, name .. "/loss", dtype)
end

-- Pinball / quantile loss, matching tfa.losses.pinball_loss.  The default
-- reduction is the mean over every element; "none" keeps the elementwise
-- values and "sum" totals them.
function Losses.pinball(graph, labels, predictions, options)
  options = options or {}
  local name = options.name or "pinball_loss"
  local dtype = options.dtype or "float32"
  labels = tf.cast(graph, labels, labels.dtype or "float32", dtype,
    name .. "/labels")
  predictions = tf.cast(graph, predictions, predictions.dtype or "float32",
    dtype, name .. "/predictions")
  local tau_value = options.tau
  if tau_value == nil then tau_value = 0.5 end
  local tau
  if type(tau_value) == "number" then
    if tau_value ~= tau_value or tau_value < 0 or tau_value > 1 then
      error("pinball tau must be a number in [0, 1]", 4)
    end
    tau = tf.constant(graph, tau_value, name .. "/tau", dtype)
  else
    tau = tf.cast(graph, tau_value, tau_value.dtype or dtype, dtype,
      name .. "/tau")
  end
  local delta = tf.sub(graph, labels, predictions, name .. "/delta", dtype)
  local one = tf.constant(graph, 1, name .. "/one", dtype)
  local positive = tf.mul(graph, tau, delta, name .. "/positive", dtype)
  local tau_minus_one = tf.sub(graph, tau, one, name .. "/tau_minus_one",
    dtype)
  local negative = tf.mul(graph, tau_minus_one, delta,
    name .. "/negative", dtype)
  local values = Ops.maximum(graph, positive, negative,
    name .. "/loss_values", dtype)
  return reduce_elementwise(graph, values, options, name, dtype, "mean")
end

Losses.PinballLoss = Losses.pinball
Losses.pinball_loss = Losses.pinball

-- Sigmoid focal cross entropy, matching tfa.losses.sigmoid_focal_crossentropy.
-- The default reduction is "none", like the TensorFlow Addons function;
-- "mean" and "sum" are also accepted.
function Losses.sigmoid_focal_crossentropy(graph, labels, predictions, options)
  options = options or {}
  local name = options.name or "sigmoid_focal_crossentropy"
  local dtype = options.dtype or "float32"
  local alpha = options.alpha
  if alpha == nil then alpha = 0.25 end
  if alpha ~= false and (type(alpha) ~= "number" or alpha ~= alpha or
      alpha < 0 or alpha > 1) then
    error("sigmoid_focal_crossentropy alpha must be a number in [0, 1]", 4)
  end
  local gamma = options.gamma
  if gamma == nil then gamma = 2 end
  if gamma ~= false and (type(gamma) ~= "number" or gamma ~= gamma or
      gamma < 0) then
    error("sigmoid_focal_crossentropy gamma must be a non-negative number", 4)
  end
  local from_logits = options.from_logits == true
  labels = tf.cast(graph, labels, labels.dtype or "float32", dtype,
    name .. "/labels")
  predictions = tf.cast(graph, predictions, predictions.dtype or "float32",
    dtype, name .. "/predictions")
  local one = tf.constant(graph, 1, name .. "/one", dtype)
  local negative_one = tf.constant(graph, -1, name .. "/negative_one", dtype)
  local binary_cross_entropy, prediction_probability
  if from_logits then
    local zero = tf.constant(graph, 0, name .. "/zero", dtype)
    local positive_logits = Ops.maximum(graph, predictions, zero,
      name .. "/positive_logits", dtype)
    local absolute_logits = tf.abs(graph, predictions,
      name .. "/absolute_logits", dtype)
    local negative_absolute = tf.mul(graph, negative_one, absolute_logits,
      name .. "/negative_absolute", dtype)
    local softplus = tf.nn.softplus(graph, negative_absolute,
      name .. "/softplus", dtype)
    local weighted_logits = tf.mul(graph, predictions, labels,
      name .. "/weighted_logits", dtype)
    binary_cross_entropy = tf.add(graph, tf.sub(graph, positive_logits,
      weighted_logits, name .. "/logit_difference", dtype), softplus,
      name .. "/cross_entropy", dtype)
    prediction_probability = tf.nn.sigmoid(graph, predictions,
      name .. "/sigmoid", dtype)
  else
    local clipped = tf.clip_by_value(graph, predictions, 1e-7, 1 - 1e-7,
      name .. "/clipped", dtype)
    local log_positive = tf.log(graph, clipped, name .. "/log_positive", dtype)
    local one_minus_labels = tf.sub(graph, one, labels,
      name .. "/one_minus_labels", dtype)
    local one_minus_clipped = tf.sub(graph, one, clipped,
      name .. "/one_minus_clipped", dtype)
    local log_negative = tf.log(graph, one_minus_clipped,
      name .. "/log_negative", dtype)
    local positive = tf.mul(graph, labels, log_positive,
      name .. "/positive", dtype)
    local negative = tf.mul(graph, one_minus_labels, log_negative,
      name .. "/negative", dtype)
    binary_cross_entropy = tf.mul(graph, negative_one,
      tf.add(graph, positive, negative, name .. "/log_sum", dtype),
      name .. "/cross_entropy", dtype)
    prediction_probability = predictions
  end
  local one_minus_labels = tf.sub(graph, one, labels,
    name .. "/probability_one_minus_labels", dtype)
  local positive_probability = tf.mul(graph, labels, prediction_probability,
    name .. "/positive_probability", dtype)
  local one_minus_probability = tf.sub(graph, one, prediction_probability,
    name .. "/one_minus_probability", dtype)
  local negative_probability = tf.mul(graph, one_minus_labels,
    one_minus_probability, name .. "/negative_probability", dtype)
  local p_t = tf.add(graph, positive_probability, negative_probability,
    name .. "/p_t", dtype)
  local loss = binary_cross_entropy
  if alpha ~= false then
    local alpha_op = tf.constant(graph, alpha, name .. "/alpha", dtype)
    local one_minus_alpha = tf.constant(graph, 1 - alpha,
      name .. "/one_minus_alpha", dtype)
    local alpha_factor = tf.add(graph, tf.mul(graph, labels, alpha_op,
      name .. "/alpha_positive", dtype), tf.mul(graph, one_minus_labels,
      one_minus_alpha, name .. "/alpha_negative", dtype),
      name .. "/alpha_factor", dtype)
    loss = tf.mul(graph, loss, alpha_factor, name .. "/alpha_loss", dtype)
  end
  if gamma ~= false and gamma ~= 0 then
    local gamma_op = tf.constant(graph, gamma, name .. "/gamma", dtype)
    local one_minus_p_t = tf.sub(graph, one, p_t,
      name .. "/one_minus_p_t", dtype)
    local modulating = Ops.pow(graph, one_minus_p_t, gamma_op,
      name .. "/modulating_factor", dtype)
    loss = tf.mul(graph, loss, modulating, name .. "/focal_loss", dtype)
  end
  return reduce_elementwise(graph, loss, options, name, dtype, "none")
end

Losses.SigmoidFocalCrossEntropy = Losses.sigmoid_focal_crossentropy

function Losses.triplet(graph, anchor, positive, negative, options)
  options = options or {}
  local name = options.name or "triplet_loss"
  local dtype = options.dtype or "float32"
  local margin = options.margin or 1
  local anchor_flat = Ops.flatten(graph, anchor, name .. "/anchor_flat", dtype)
  local positive_flat = Ops.flatten(graph, positive, name .. "/positive_flat",
    dtype)
  local negative_flat = Ops.flatten(graph, negative, name .. "/negative_flat",
    dtype)
  local positive_delta = tf.sub(graph, anchor_flat, positive_flat,
    name .. "/positive_delta", dtype)
  local negative_delta = tf.sub(graph, anchor_flat, negative_flat,
    name .. "/negative_delta", dtype)
  local positive_distance = tf.sqrt(graph, Ops.sum_all(graph,
    tf.square(graph, positive_delta, name .. "/positive_square", dtype),
    name .. "/positive_distance", dtype), name .. "/positive_norm", dtype)
  local negative_distance = tf.sqrt(graph, Ops.sum_all(graph,
    tf.square(graph, negative_delta, name .. "/negative_square", dtype),
    name .. "/negative_distance", dtype), name .. "/negative_norm", dtype)
  local difference = tf.add(graph, tf.sub(graph, positive_distance,
    negative_distance, name .. "/distance_difference", dtype),
    tf.constant(graph, margin, name .. "/margin", dtype),
    name .. "/margin_difference", dtype)
  return Ops.maximum(graph, difference, tf.constant(graph, 0,
    name .. "/zero", dtype), name .. "/loss", dtype)
end

-- N-pairs loss, matching tfa.losses.npairs_loss.  The prediction is the
-- [batch_size, batch_size] similarity matrix; labels are 1-D class ids.
function Losses.npairs_loss(graph, labels, similarity, options)
  options = options or {}
  local name = options.name or "npairs_loss"
  local dtype = options.dtype or similarity.dtype or "float32"
  local precision = similarity.dtype or "float32"
  similarity = tf.cast(graph, similarity, precision, dtype,
    name .. "/similarity")
  local label_dtype = labels.dtype or "int32"
  labels = tf.cast(graph, labels, label_dtype, "int32", name .. "/labels")
  local column = tf.reshape(graph, labels, { -1, 1 }, name .. "/column",
    "int32")
  local row = tf.reshape(graph, labels, { 1, -1 }, name .. "/row", "int32")
  local same = Ops.equal(graph, column, row, name .. "/same", "int32")
  local adjacency = tf.cast(graph, same, "bool", dtype, name .. "/adjacency")
  return npairs_common(graph, similarity, adjacency, name, dtype)
end

Losses.NpairsLoss = Losses.npairs_loss

-- Multilabel N-pairs loss, matching tfa.losses.npairs_multilabel_loss.  Labels
-- are [batch_size, num_classes] binary indicators; the adjacency matrix counts
-- shared labels and each row is normalized before the softmax cross entropy.
function Losses.npairs_multilabel_loss(graph, labels, similarity, options)
  options = options or {}
  local name = options.name or "npairs_multilabel_loss"
  local dtype = options.dtype or similarity.dtype or "float32"
  local precision = similarity.dtype or "float32"
  similarity = tf.cast(graph, similarity, precision, dtype,
    name .. "/similarity")
  labels = tf.cast(graph, labels, labels.dtype or dtype, dtype,
    name .. "/labels")
  local counts = matmul(graph, labels, labels, false, true,
    name .. "/counts", dtype)
  return npairs_common(graph, similarity, counts, name, dtype)
end

Losses.NpairsMultilabelLoss = Losses.npairs_multilabel_loss

local function pairwise_distance(graph, embeddings, metric, name, dtype)
  local normal = string.lower(metric or "L2")
  if normal == "angular" then
    local squared = tf.square(graph, embeddings, name .. "/squared", dtype)
    local sums = tf.reduce_sum(graph, squared, { 1 }, name .. "/sums", dtype,
      true)
    local floor = tf.constant(graph, 1e-12, name .. "/angular_epsilon",
      dtype)
    local safe = Ops.maximum(graph, sums, floor, name .. "/angular_safe",
      dtype)
    local norm = tf.sqrt(graph, safe, name .. "/norm", dtype)
    local normalized = tf.div(graph, embeddings, norm,
      name .. "/normalized", dtype)
    local similarity = matmul(graph, normalized, normalized, false, true,
      name .. "/similarity", dtype)
    local one = tf.constant(graph, 1, name .. "/angular_one", dtype)
    local zero = tf.constant(graph, 0, name .. "/angular_zero", dtype)
    return Ops.maximum(graph, tf.sub(graph, one, similarity,
      name .. "/angular_raw", dtype), zero, name .. "/angular", dtype)
  end

  local squared = tf.square(graph, embeddings, name .. "/squared", dtype)
  local row_sums = tf.reduce_sum(graph, squared, { 1 }, name .. "/row_sums",
    dtype, true)
  local transposed = tf.transpose(graph, embeddings, { 1, 0 },
    name .. "/transposed", dtype)
  local transposed_squared = tf.square(graph, transposed,
    name .. "/transposed_squared", dtype)
  local column_sums = tf.reduce_sum(graph, transposed_squared, { 0 },
    name .. "/column_sums", dtype, true)
  local products = matmul(graph, embeddings, transposed, false, false,
    name .. "/products", dtype)
  local two = tf.constant(graph, 2, name .. "/distance_two", dtype)
  local zero = tf.constant(graph, 0, name .. "/distance_zero", dtype)
  local distances_squared = tf.sub(graph, tf.add(graph, row_sums, column_sums,
    name .. "/sum_squares", dtype), tf.mul(graph, two, products,
    name .. "/twice_products", dtype), name .. "/difference", dtype)
  distances_squared = Ops.maximum(graph, distances_squared, zero,
    name .. "/squared_distances", dtype)
  local error_mask = Ops.less_equal(graph, distances_squared, zero,
    name .. "/error_mask", dtype)
  local distances
  if normal == "squared-l2" then
    distances = distances_squared
  elseif normal == "l2" then
    local error_float = tf.cast(graph, error_mask, "bool", dtype,
      name .. "/error_float")
    local epsilon = tf.constant(graph, 1e-16, name .. "/distance_epsilon",
      dtype)
    local stable = tf.add(graph, distances_squared, tf.mul(graph, error_float,
      epsilon, name .. "/stable_offset", dtype), name .. "/stable", dtype)
    distances = tf.sqrt(graph, stable, name .. "/distances", dtype)
  else
    error("triplet_semihard_loss distance_metric must be 'L2', " ..
      "'squared-L2', or 'angular'", 4)
  end
  local valid = tf.cast(graph, Ops.logical_not(graph, error_mask,
    name .. "/valid_mask"), "bool", dtype, name .. "/valid")
  distances = tf.mul(graph, distances, valid, name .. "/masked", dtype)
  local batch_size = dimension(graph, embeddings, 0,
    name .. "/pairwise_batch_size", dtype)
  local identity = identity_matrix(graph, batch_size,
    name .. "/distance_identity", dtype)
  local one = tf.constant(graph, 1, name .. "/distance_one", dtype)
  local off_diagonals = tf.sub(graph, one, identity,
    name .. "/off_diagonals", dtype)
  return tf.mul(graph, distances, off_diagonals, name .. "/pairwise", dtype)
end

local function masked_minimum(graph, data, mask, name, dtype)
  local maximum = tf.reduce_max(graph, data, { 1 }, name .. "/maximum", dtype,
    true)
  local shifted = tf.sub(graph, data, maximum, name .. "/shifted", dtype)
  local masked = tf.mul(graph, shifted, mask, name .. "/masked", dtype)
  local minimum = tf.reduce_min(graph, masked, { 1 }, name .. "/minimum",
    dtype, true)
  return tf.add(graph, minimum, maximum, name, dtype)
end

local function masked_maximum(graph, data, mask, name, dtype)
  local minimum = tf.reduce_min(graph, data, { 1 }, name .. "/minimum", dtype,
    true)
  local shifted = tf.sub(graph, data, minimum, name .. "/shifted", dtype)
  local masked = tf.mul(graph, shifted, mask, name .. "/masked", dtype)
  local maximum = tf.reduce_max(graph, masked, { 1 }, name .. "/maximum",
    dtype, true)
  return tf.add(graph, maximum, minimum, name, dtype)
end

-- Semi-hard triplet loss, matching tfa.losses.triplet_semihard_loss.  Labels
-- are 1-D class ids and predictions are [batch_size, embedding_dim] vectors.
function Losses.triplet_semihard_loss(graph, labels, embeddings, options)
  options = options or {}
  local name = options.name or "triplet_semihard_loss"
  local source_dtype = options.dtype or embeddings.dtype or "float32"
  local dtype = source_dtype
  local restore = false
  if dtype == "float16" or dtype == "bfloat16" then
    dtype = "float32"
    restore = true
  end
  local margin = options.margin or 1
  if type(margin) ~= "number" or margin ~= margin then
    error("triplet_semihard_loss margin must be a number", 4)
  end
  local metric = options.distance_metric or "L2"
  if type(metric) ~= "string" then
    error("triplet_semihard_loss distance_metric must be a string", 4)
  end
  embeddings = tf.cast(graph, embeddings, source_dtype, dtype,
    name .. "/embeddings")
  labels = tf.cast(graph, labels, labels.dtype or "int32", "int32",
    name .. "/labels")

  local distances = pairwise_distance(graph, embeddings, metric, name, dtype)
  local column = tf.reshape(graph, labels, { -1, 1 }, name .. "/column",
    "int32")
  local row = tf.reshape(graph, labels, { 1, -1 }, name .. "/row", "int32")
  local adjacency_bool = Ops.equal(graph, column, row, name .. "/adjacency",
    "int32")
  local adjacency_not_bool = Ops.logical_not(graph, adjacency_bool,
    name .. "/adjacency_not")
  local adjacency = tf.cast(graph, adjacency_bool, "bool", dtype,
    name .. "/adjacency_float")
  local adjacency_not = tf.cast(graph, adjacency_not_bool, "bool", dtype,
    name .. "/adjacency_not_float")
  local batch_size = dimension(graph, embeddings, 0, name .. "/batch_size",
    dtype)
  local zip = tf.constant(graph, 1, name .. "/tile_one", "int32")
  local zero = tf.constant(graph, 0, name .. "/zero", dtype)

  local distance_tile = tile_dynamic(graph, distances, { batch_size, zip },
    name .. "/distance_tile", dtype)
  local adjacency_not_tile = tile_dynamic(graph, adjacency_not_bool,
    { batch_size, zip }, name .. "/adjacency_not_tile", "bool")
  local transposed_distances = tf.transpose(graph, distances, { 1, 0 },
    name .. "/transposed_distances", dtype)
  local distance_column = tf.reshape(graph, transposed_distances, { -1, 1 },
    name .. "/distance_column", dtype)
  local mask = Ops.logical_and(graph, adjacency_not_tile,
    Ops.greater(graph, distance_tile, distance_column,
      name .. "/greater_distance", dtype), name .. "/mask")
  local mask_float = tf.cast(graph, mask, "bool", dtype, name .. "/mask_float")
  local mask_counts = tf.reduce_sum(graph, mask_float, { 1 },
    name .. "/mask_counts", dtype, true)
  local has_mask = Ops.greater(graph, mask_counts, zero,
    name .. "/has_mask", dtype)
  local mask_final = tf.transpose(graph, reshape_dynamic(graph, has_mask,
    { batch_size, batch_size }, name .. "/mask_final_flat", "bool"),
    { 1, 0 }, name .. "/mask_final", "bool")

  local negatives_outside = masked_minimum(graph, distance_tile, mask_float,
    name .. "/negatives_outside_flat", dtype)
  negatives_outside = transpose(graph, reshape_dynamic(graph,
    negatives_outside, { batch_size, batch_size },
    name .. "/negatives_outside_reshape", dtype), { 1, 0 },
    name .. "/negatives_outside", dtype)
  local negatives_inside = masked_maximum(graph, distances, adjacency_not,
    name .. "/negatives_inside_flat", dtype)
  negatives_inside = tile_dynamic(graph, negatives_inside, { zip, batch_size },
    name .. "/negatives_inside", dtype)
  local semi_hard = tf.where(graph, mask_final, negatives_outside,
    negatives_inside, name .. "/semi_hard", dtype)

  local margin_op = tf.constant(graph, margin, name .. "/margin", dtype)
  local loss_matrix = tf.add(graph, tf.sub(graph, distances, semi_hard,
    name .. "/distance_difference", dtype), margin_op,
    name .. "/loss_matrix", dtype)
  local identity = identity_matrix(graph, batch_size, name .. "/identity",
    dtype)
  local positive_mask = tf.sub(graph, adjacency, identity,
    name .. "/positive_mask", dtype)
  local positive_loss = Ops.maximum(graph, tf.mul(graph, loss_matrix,
    positive_mask, name .. "/positive_loss_values", dtype), zero,
    name .. "/positive_loss", dtype)
  local numerator = Ops.sum_all(graph, positive_loss,
    name .. "/positive_sum", dtype)
  local denominator = Ops.sum_all(graph, positive_mask,
    name .. "/positive_count", dtype)
  local loss = Ops.clean_div(graph, numerator, denominator, name, dtype, 1e-12)
  if restore then
    loss = tf.cast(graph, loss, dtype, source_dtype, name .. "/restored")
  end
  return loss
end

Losses.TripletSemiHardLoss = Losses.triplet_semihard_loss

return Losses
