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
