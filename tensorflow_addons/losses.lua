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

return Losses
