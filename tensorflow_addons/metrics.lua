local tf = require("tensorflow")
local Ops = require("tensorflow_addons.ops")

local Metrics = {}

local function binary_predictions(graph, predictions, labels, options, name)
  local dtype = options.dtype or "float32"
  local threshold = options.threshold or 0.5
  labels = tf.cast(graph, labels, labels.dtype or "float32", dtype,
    name .. "/labels")
  local threshold_op = tf.constant(graph, threshold, name .. "/threshold", dtype)
  local predicted = Ops.greater(graph, predictions, threshold_op,
    name .. "/greater", dtype)
  return tf.cast(graph, predicted, "bool", dtype, name .. "/predicted"), labels
end

local function binary_counts(graph, predictions, labels, threshold, name)
  local predicted, labels_float = binary_predictions(graph, predictions, labels,
    { threshold = threshold }, name)
  local one = tf.constant(graph, 1, name .. "/one", "float32")
  local tp = Ops.sum_all(graph, tf.mul(graph, predicted, labels_float,
    name .. "/tp_tensor", "float32"), name .. "/tp", "float32")
  local fp = Ops.sum_all(graph, tf.mul(graph, predicted,
    tf.sub(graph, one, labels_float, name .. "/one_minus_labels", "float32"),
    name .. "/fp_tensor", "float32"), name .. "/fp", "float32")
  local fn = Ops.sum_all(graph, tf.mul(graph,
    tf.sub(graph, one, predicted, name .. "/one_minus_predicted", "float32"),
    labels_float, name .. "/fn_tensor", "float32"), name .. "/fn", "float32")
  local tn = Ops.sum_all(graph, tf.mul(graph,
    tf.sub(graph, one, predicted, name .. "/one_minus_predicted_tn", "float32"),
    tf.sub(graph, one, labels_float, name .. "/one_minus_labels_tn", "float32"),
    name .. "/tn_tensor", "float32"), name .. "/tn", "float32")
  return tp, fp, fn, tn
end

local function class_id_counts(graph, predictions, labels, options, name)
  local class_id = options.class_id
  if type(class_id) ~= "number" or class_id % 1 ~= 0 or class_id < 0 then
    error(name .. " class_id must be a non-negative integer", 4)
  end
  local dtype = options.dtype or "float32"
  local index = tf.constant(graph, class_id, name .. "/class_id", "int32")
  local prediction_class = tf.gather(graph, predictions, index, -1,
    name .. "/prediction_class", predictions.dtype or dtype)
  local label_class = tf.gather(graph, labels, index, -1,
    name .. "/label_class", labels.dtype or dtype)
  return binary_counts(graph, prediction_class, label_class,
    options.threshold or 0.5, name)
end

local function finite_number(value)
  return type(value) == "number" and value == value and value ~= math.huge and
    value ~= -math.huge
end

local function validate_threshold_count(value, name)
  if type(value) ~= "number" or value % 1 ~= 0 or value < 1 then
    error(name .. " num_thresholds must be a positive integer", 4)
  end
  return value
end

-- Keras-style thresholds for the sensitivity/specificity metrics: a linearly
-- spaced set in [0, 1] with both endpoints.  A single threshold is 0.5.
local function sensitivity_thresholds(count, name)
  count = validate_threshold_count(count or 200, name)
  if count == 1 then return { 0.5 } end
  local values = { 0 }
  for index = 1, count - 2 do
    values[#values + 1] = index / (count - 1)
  end
  values[#values + 1] = 1
  return values
end

-- AUC thresholds follow Keras: the interior thresholds are evenly spaced in
-- (0, 1), and endpoints slightly outside [0, 1] handle exact 0 and 1 scores.
local function auc_thresholds(options, name)
  local epsilon = 1e-7
  local supplied = options.thresholds
  if supplied ~= nil then
    if type(supplied) ~= "table" or #supplied == 0 then
      error(name .. " thresholds must be a non-empty list", 4)
    end
    local values = {}
    for index, value in ipairs(supplied) do
      if not finite_number(value) or value < 0 or value > 1 then
        error(name .. " thresholds must be numbers in [0, 1]", 4)
      end
      values[index] = value
    end
    table.sort(values)
    local result = { -epsilon }
    for _, value in ipairs(values) do result[#result + 1] = value end
    result[#result + 1] = 1 + epsilon
    return result, #result
  end
  local count = validate_threshold_count(options.num_thresholds or 200, name)
  if count <= 1 then
    error(name .. " num_thresholds must be greater than 1", 4)
  end
  local values = { -epsilon }
  for index = 1, count - 2 do
    values[#values + 1] = index / (count - 1)
  end
  values[#values + 1] = 1 + epsilon
  return values, count
end

local function vector_slice(graph, input, start, length, name, dtype)
  return tf.slice(graph, input, { start }, { length }, name, dtype)
end

-- Count TP/FP/FN/TN at every threshold.  Labels and predictions are flattened,
-- matching Keras' default binary (non-multi-label) treatment.
local function thresholded_counts(graph, predictions, labels, thresholds,
    from_logits, name, dtype)
  labels = tf.cast(graph, labels, labels.dtype or dtype, dtype,
    name .. "/labels")
  predictions = tf.cast(graph, predictions, predictions.dtype or dtype, dtype,
    name .. "/predictions")
  if from_logits then
    predictions = tf.nn.sigmoid(graph, predictions, name .. "/sigmoid", dtype)
  end
  local flat_labels = Ops.flatten(graph, labels, name .. "/labels_flat", dtype)
  local flat_predictions = Ops.flatten(graph, predictions,
    name .. "/predictions_flat", dtype)
  local threshold_op = tf.constant(graph, thresholds, name .. "/thresholds",
    dtype)
  local threshold_matrix = tf.expand_dims(graph, threshold_op, 1,
    name .. "/threshold_matrix", dtype)
  local prediction_matrix = tf.expand_dims(graph, flat_predictions, 0,
    name .. "/prediction_matrix", dtype)
  local positive = Ops.greater(graph, prediction_matrix, threshold_matrix,
    name .. "/positive", dtype)
  local positive_float = tf.cast(graph, positive, "bool", dtype,
    name .. "/positive_float")
  local label_matrix = tf.expand_dims(graph, flat_labels, 0,
    name .. "/label_matrix", dtype)
  local one = tf.constant(graph, 1, name .. "/one", dtype)
  local label_negative = tf.sub(graph, one, label_matrix,
    name .. "/label_negative", dtype)
  local prediction_negative = tf.sub(graph, one, positive_float,
    name .. "/prediction_negative", dtype)
  local true_positive = tf.reduce_sum(graph, tf.mul(graph, positive_float,
    label_matrix, name .. "/tp_tensor", dtype), { 1 }, name .. "/tp", dtype,
    false)
  local false_positive = tf.reduce_sum(graph, tf.mul(graph, positive_float,
    label_negative, name .. "/fp_tensor", dtype), { 1 }, name .. "/fp", dtype,
    false)
  local false_negative = tf.reduce_sum(graph, tf.mul(graph, prediction_negative,
    label_matrix, name .. "/fn_tensor", dtype), { 1 }, name .. "/fn", dtype,
    false)
  local true_negative = tf.reduce_sum(graph, tf.mul(graph, prediction_negative,
    label_negative, name .. "/tn_tensor", dtype), { 1 }, name .. "/tn", dtype,
    false)
  return true_positive, false_positive, false_negative, true_negative
end

local function true_positive_rate(graph, true_positive, false_negative, name,
    dtype)
  return Ops.clean_div(graph, true_positive, tf.add(graph, true_positive,
    false_negative, name .. "/actual_positive", dtype), name, dtype)
end

local function max_under_constraint(graph, constrained, dependent, target, name,
    dtype)
  local feasible = tf.greater_equal(graph, constrained,
    tf.constant(graph, target, name .. "/target", dtype),
    name .. "/feasible", dtype)
  local feasible_float = tf.cast(graph, feasible, "bool", dtype,
    name .. "/feasible_float")
  local masked = tf.mul(graph, dependent, feasible_float,
    name .. "/masked", dtype)
  local maximum = tf.reduce_max(graph, masked, { 0 }, name .. "/maximum", dtype,
    false)
  return Ops.maximum(graph, maximum, tf.constant(graph, 0, name .. "/zero",
    dtype), name, dtype)
end

local integer_dtypes = {
  int8 = true,
  int16 = true,
  int32 = true,
  int64 = true,
  uint8 = true,
  uint16 = true,
  uint32 = true,
  uint64 = true,
}

local function tensor_dtype(tensor, fallback)
  return (type(tensor) == "table" and tensor.dtype) or fallback or "float32"
end

local function is_integer_dtype(dtype)
  return integer_dtypes[dtype] == true
end

local function cast_to_int64(graph, tensor, name)
  local dtype = tensor_dtype(tensor)
  if dtype == "int64" then return tensor end
  return tf.cast(graph, tensor, dtype, "int64", name)
end

-- Convert labels or predictions to integer class ids.  Integer tensors are
-- already class ids; floating point tensors are one-hot rows or score vectors.
local function tensor_classes(graph, tensor, options, one_hot_option, name)
  local dtype = tensor_dtype(tensor, options.dtype)
  local one_hot = options[one_hot_option]
  if one_hot == nil then one_hot = not is_integer_dtype(dtype) end
  if one_hot then
    return tf.argmax(graph, tensor, -1, name .. "/argmax", dtype, "int64"),
      true
  end
  return cast_to_int64(graph, tensor, name .. "/cast"), false
end

local function last_dimension(graph, tensor, name, dtype)
  local shape = tf.shape(graph, tensor, name .. "/input_shape",
    dtype or tensor_dtype(tensor))
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
  local sliced = desc:finish()
  return tf.reshape(graph, sliced, {}, name .. "/scalar", "int32")
end

local function class_count(graph, predictions, labels, pred_one_hot,
    label_one_hot, options, name)
  if options.num_classes ~= nil then
    return tf.constant(graph, options.num_classes, name .. "/count", "int32")
  end
  if label_one_hot then
    return last_dimension(graph, labels, name .. "/label_dimension")
  end
  if pred_one_hot then
    return last_dimension(graph, predictions, name .. "/pred_dimension")
  end
  local label_classes = cast_to_int64(graph, labels, name .. "/label_int")
  local pred_classes = cast_to_int64(graph, predictions, name .. "/pred_int")
  local label_max = tf.reduce_max(graph, label_classes, { 0 },
    name .. "/label_max", "int64", false)
  local pred_max = tf.reduce_max(graph, pred_classes, { 0 },
    name .. "/pred_max", "int64", false)
  local maximum = tf.maximum(graph, label_max, pred_max, name .. "/maximum",
    "int64")
  local one = tf.constant(graph, 1, name .. "/one", "int64")
  local two = tf.constant(graph, 2, name .. "/two", "int64")
  local count = tf.maximum(graph, tf.add(graph, maximum, one,
    name .. "/plus_one", "int64"), two, name .. "/at_least_two", "int64")
  return tf.cast(graph, count, "int64", "int32", name .. "/count")
end

local function one_hot_classes(graph, indices, depth, name)
  local depth_op = depth
  if type(depth) == "number" then
    depth_op = tf.constant(graph, depth, name .. "/depth", "int32")
  end
  local desc = graph:new_operation("OneHot", name)
  desc:add_input(indices)
  desc:add_input(depth_op)
  desc:add_input(tf.constant(graph, 1, name .. "/on", "float32"))
  desc:add_input(tf.constant(graph, 0, name .. "/off", "float32"))
  desc:set_type("T", "float32")
  desc:set_type("TI", "int64")
  desc:set_int("axis", -1)
  return desc:finish()
end

local function validate_num_classes(value, name)
  if value == nil then return end
  if type(value) ~= "number" or value % 1 ~= 0 or value < 2 then
    error(name .. " requires num_classes to be an integer >= 2", 4)
  end
end

local function multiclass_counts(graph, predictions, labels, options, name)
  local pred_classes, pred_one_hot = tensor_classes(graph, predictions,
    options, "predictions_one_hot", name .. "/predictions")
  local label_classes, label_one_hot = tensor_classes(graph, labels, options,
    "labels_one_hot", name .. "/labels")
  local count = class_count(graph, predictions, labels, pred_one_hot,
    label_one_hot, options, name .. "/classes")
  local pred_hot = one_hot_classes(graph, pred_classes, count,
    name .. "/predicted_one_hot")
  local label_hot = one_hot_classes(graph, label_classes, count,
    name .. "/labels_one_hot")
  local pred_support = tf.reduce_sum(graph, pred_hot, { 0 },
    name .. "/predicted_support", "float32", false)
  local label_support = tf.reduce_sum(graph, label_hot, { 0 },
    name .. "/label_support", "float32", false)
  local true_positive = tf.reduce_sum(graph, tf.mul(graph, pred_hot,
    label_hot, name .. "/true_positive_tensor", "float32"), { 0 },
    name .. "/true_positive", "float32", false)
  local false_positive = tf.sub(graph, pred_support, true_positive,
    name .. "/false_positive", "float32")
  local false_negative = tf.sub(graph, label_support, true_positive,
    name .. "/false_negative", "float32")
  return true_positive, false_positive, false_negative, label_support
end

local function multiclass_average_score(graph, predictions, labels, options,
    name, kind)
  local average = options.average or "macro"
  if average ~= "macro" and average ~= "micro" and average ~= "weighted" then
    error(name .. " average must be 'macro', 'micro', or 'weighted'", 4)
  end
  local true_positive, false_positive, false_negative, label_support =
    multiclass_counts(graph, predictions, labels, options, name)
  if average == "micro" then
    true_positive = Ops.sum_all(graph, true_positive, name .. "/tp_total",
      "float32")
    false_positive = Ops.sum_all(graph, false_positive, name .. "/fp_total",
      "float32")
    false_negative = Ops.sum_all(graph, false_negative, name .. "/fn_total",
      "float32")
  end
  local two = tf.constant(graph, 2, name .. "/two", "float32")
  local per_class
  if kind == "f1" then
    per_class = Ops.clean_div(graph, tf.mul(graph, two, true_positive,
      name .. "/numerator", "float32"), tf.add(graph, tf.mul(graph, two,
      true_positive, name .. "/two_tp", "float32"), tf.add(graph,
      false_positive, false_negative, name .. "/errors", "float32"),
      name .. "/denominator", "float32"), name .. "/per_class", "float32")
  elseif kind == "precision" then
    per_class = Ops.clean_div(graph, true_positive, tf.add(graph,
      true_positive, false_positive, name .. "/predicted_positive", "float32"),
      name .. "/per_class", "float32")
  elseif kind == "recall" then
    per_class = Ops.clean_div(graph, true_positive, tf.add(graph,
      true_positive, false_negative, name .. "/actual_positive", "float32"),
      name .. "/per_class", "float32")
  else
    error("unknown multiclass score kind " .. tostring(kind), 4)
  end
  if average == "macro" then
    return tf.reduce_mean(graph, per_class, { 0 }, name, "float32", false)
  end
  if average == "micro" then return per_class end
  local weighted = tf.mul(graph, per_class, label_support,
    name .. "/weighted", "float32")
  return Ops.clean_div(graph, Ops.sum_all(graph, weighted,
    name .. "/weighted_sum", "float32"), Ops.sum_all(graph, label_support,
    name .. "/support_sum", "float32"), name, "float32")
end

local function multiclass_f1(graph, predictions, labels, options, name)
  return multiclass_average_score(graph, predictions, labels, options, name,
    "f1")
end

local function is_multiclass_request(options)
  return options.average ~= nil or options.num_classes ~= nil or
    options.predictions_one_hot ~= nil or options.labels_one_hot ~= nil
end

-- Area under the ROC or precision-recall curve for binary predictions.  The
-- threshold sweep, endpoint epsilon, and summation methods follow
-- tf.keras.metrics.AUC; predictions are flattened into one label dimension.
function Metrics.auc(graph, predictions, labels, options)
  options = options or {}
  local name = options.name or "auc"
  local dtype = options.dtype or "float32"
  local curve = string.upper(options.curve or "ROC")
  if curve ~= "ROC" and curve ~= "PR" then
    error(name .. " curve must be 'ROC' or 'PR'", 4)
  end
  local summation = string.upper(options.summation_method or "interpolation")
  if summation ~= "INTERPOLATION" and summation ~= "MINORING" and
      summation ~= "MAJORING" then
    error(name .. " summation_method must be 'interpolation', 'minoring', " ..
      "or 'majoring'", 4)
  end
  local thresholds, count = auc_thresholds(options, name)
  local tp, fp, fn, tn = thresholded_counts(graph, predictions, labels,
    thresholds, options.from_logits == true, name, dtype)
  local recall = true_positive_rate(graph, tp, fn, name .. "/recall", dtype)

  if curve == "PR" and summation == "INTERPOLATION" then
    local length = count - 1
    local tp_previous = vector_slice(graph, tp, 0, length,
      name .. "/tp_previous", dtype)
    local tp_next = vector_slice(graph, tp, 1, length, name .. "/tp_next",
      dtype)
    local fp_previous = vector_slice(graph, fp, 0, length,
      name .. "/fp_previous", dtype)
    local fp_next = vector_slice(graph, fp, 1, length, name .. "/fp_next",
      dtype)
    local fn_next = vector_slice(graph, fn, 1, length, name .. "/fn_next",
      dtype)
    local predicted_previous = tf.add(graph, tp_previous, fp_previous,
      name .. "/predicted_previous", dtype)
    local predicted_next = tf.add(graph, tp_next, fp_next,
      name .. "/predicted_next", dtype)
    local dtp = tf.sub(graph, tp_previous, tp_next, name .. "/dtp", dtype)
    local dp = tf.sub(graph, predicted_previous, predicted_next,
      name .. "/dp", dtype)
    local zero = tf.constant(graph, 0, name .. "/pr_zero", dtype)
    local one = tf.constant(graph, 1, name .. "/pr_one", dtype)
    local slope = Ops.clean_div(graph, dtp, Ops.maximum(graph, dp, zero,
      name .. "/dp_safe", dtype), name .. "/prec_slope", dtype)
    local intercept = tf.sub(graph, tp_next, tf.mul(graph, slope,
      predicted_next, name .. "/slope_times_predicted", dtype),
      name .. "/intercept", dtype)
    local both_positive = tf.logical_and(graph, Ops.greater(graph,
      predicted_previous, zero, name .. "/previous_positive", dtype),
      Ops.greater(graph, predicted_next, zero, name .. "/next_positive", dtype),
      name .. "/both_positive")
    local ratio = Ops.clean_div(graph, predicted_previous,
      Ops.maximum(graph, predicted_next, zero, name .. "/predicted_next_safe",
        dtype), name .. "/ratio", dtype)
    local ones = tf.add(graph, tf.mul(graph, predicted_next, zero,
      name .. "/ones_base", dtype), one, name .. "/ones", dtype)
    local safe_ratio = tf.where(graph, both_positive, ratio, ones,
      name .. "/safe_ratio", dtype)
    local log_ratio = tf.log(graph, safe_ratio, name .. "/log_ratio", dtype)
    local numerator = tf.mul(graph, slope, tf.add(graph, dtp,
      tf.mul(graph, intercept, log_ratio, name .. "/intercept_term", dtype),
      name .. "/increment_numerator", dtype), name .. "/increment_numerator_scaled",
      dtype)
    local denominator = Ops.maximum(graph, tf.add(graph, tp_next, fn_next,
      name .. "/positive_next", dtype), zero, name .. "/positive_next_safe",
      dtype)
    local increments = Ops.clean_div(graph, numerator, denominator,
      name .. "/pr_increment", dtype)
    return Ops.sum_all(graph, increments, name, dtype)
  end

  local x, y
  if curve == "ROC" then
    x = Ops.clean_div(graph, fp, tf.add(graph, fp, tn,
      name .. "/actual_negative", dtype), name .. "/false_positive_rate", dtype)
    y = recall
  else
    x = recall
    y = Ops.clean_div(graph, tp, tf.add(graph, tp, fp,
      name .. "/predicted_positive", dtype), name .. "/precision", dtype)
  end
  local length = count - 1
  local x_previous = vector_slice(graph, x, 0, length,
    name .. "/x_previous", dtype)
  local x_next = vector_slice(graph, x, 1, length, name .. "/x_next", dtype)
  local y_previous = vector_slice(graph, y, 0, length,
    name .. "/y_previous", dtype)
  local y_next = vector_slice(graph, y, 1, length, name .. "/y_next", dtype)
  local heights
  if summation == "INTERPOLATION" then
    heights = tf.div(graph, tf.add(graph, y_previous, y_next,
      name .. "/height_sum", dtype), tf.constant(graph, 2,
      name .. "/two", dtype), name .. "/heights", dtype)
  elseif summation == "MINORING" then
    heights = Ops.minimum(graph, y_previous, y_next, name .. "/heights", dtype)
  else
    heights = Ops.maximum(graph, y_previous, y_next, name .. "/heights", dtype)
  end
  local widths = tf.sub(graph, x_previous, x_next, name .. "/widths", dtype)
  local terms = tf.mul(graph, widths, heights, name .. "/terms", dtype)
  return Ops.sum_all(graph, terms, name, dtype)
end

function Metrics.f1_score(graph, predictions, labels, options)
  options = options or {}
  local name = options.name or "f1_score"
  validate_num_classes(options.num_classes, name)
  local tp, fp, fn
  if options.class_id ~= nil then
    tp, fp, fn = class_id_counts(graph, predictions, labels, options, name)
  else
    if is_multiclass_request(options) then
      return multiclass_f1(graph, predictions, labels, options, name)
    end
    tp, fp, fn = binary_counts(graph, predictions, labels,
      options.threshold or 0.5, name)
  end
  local numerator = tf.mul(graph, tf.constant(graph, 2, name .. "/two",
    "float32"), tp, name .. "/numerator", "float32")
  local denominator = tf.add(graph, numerator, tf.add(graph, fp, fn,
    name .. "/errors", "float32"), name .. "/denominator", "float32")
  return Ops.clean_div(graph, numerator, denominator, name, "float32")
end

-- Precision, mirroring f1_score's binary/multiclass API.  Binary predictions
-- are thresholded with `options.threshold`; multiclass inputs use
-- `options.average = "macro" | "micro" | "weighted"` (default "macro").
function Metrics.precision(graph, predictions, labels, options)
  options = options or {}
  local name = options.name or "precision"
  validate_num_classes(options.num_classes, name)
  local tp, fp
  if options.class_id ~= nil then
    tp, fp = class_id_counts(graph, predictions, labels, options, name)
  else
    if is_multiclass_request(options) then
      return multiclass_average_score(graph, predictions, labels, options,
        name, "precision")
    end
    tp, fp = binary_counts(graph, predictions, labels,
      options.threshold or 0.5, name)
  end
  return Ops.clean_div(graph, tp, tf.add(graph, tp, fp,
    name .. "/predicted_positive", "float32"), name, "float32")
end

Metrics.Precision = Metrics.precision

-- Recall (sensitivity), with the same binary and multiclass options as
-- precision.
function Metrics.recall(graph, predictions, labels, options)
  options = options or {}
  local name = options.name or "recall"
  validate_num_classes(options.num_classes, name)
  local tp, fn, _
  if options.class_id ~= nil then
    tp, _, fn = class_id_counts(graph, predictions, labels, options, name)
  else
    if is_multiclass_request(options) then
      return multiclass_average_score(graph, predictions, labels, options,
        name, "recall")
    end
    tp, _, fn = binary_counts(graph, predictions, labels,
      options.threshold or 0.5, name)
  end
  return Ops.clean_div(graph, tp, tf.add(graph, tp, fn,
    name .. "/actual_positive", "float32"), name, "float32")
end

Metrics.Recall = Metrics.recall

-- Best sensitivity whose operating point has specificity >= the requested
-- value, following tf.keras.metrics.SensitivityAtSpecificity.
function Metrics.sensitivity_at_specificity(graph, predictions, labels,
    specificity, options)
  if type(specificity) == "table" and options == nil then
    options = specificity
    specificity = options.specificity
  end
  options = options or {}
  if specificity == nil then specificity = options.specificity end
  if not finite_number(specificity) or specificity < 0 or specificity > 1 then
    error("sensitivity_at_specificity specificity must be in [0, 1]", 4)
  end
  local name = options.name or "sensitivity_at_specificity"
  local dtype = options.dtype or "float32"
  local thresholds = sensitivity_thresholds(options.num_thresholds, name)
  local tp, fp, fn, tn = thresholded_counts(graph, predictions, labels,
    thresholds, options.from_logits == true, name, dtype)
  local sensitivity = true_positive_rate(graph, tp, fn,
    name .. "/sensitivity", dtype)
  local specificity_values = Ops.clean_div(graph, tn, tf.add(graph, tn, fp,
    name .. "/actual_negative", dtype), name .. "/specificity", dtype)
  return max_under_constraint(graph, specificity_values, sensitivity,
    specificity, name, dtype)
end

Metrics.SensitivityAtSpecificity = Metrics.sensitivity_at_specificity

-- Best specificity whose operating point has sensitivity >= the requested
-- value, following tf.keras.metrics.SpecificityAtSensitivity.
function Metrics.specificity_at_sensitivity(graph, predictions, labels,
    sensitivity, options)
  if type(sensitivity) == "table" and options == nil then
    options = sensitivity
    sensitivity = options.sensitivity
  end
  options = options or {}
  if sensitivity == nil then sensitivity = options.sensitivity end
  if not finite_number(sensitivity) or sensitivity < 0 or sensitivity > 1 then
    error("specificity_at_sensitivity sensitivity must be in [0, 1]", 4)
  end
  local name = options.name or "specificity_at_sensitivity"
  local dtype = options.dtype or "float32"
  local thresholds = sensitivity_thresholds(options.num_thresholds, name)
  local tp, fp, fn, tn = thresholded_counts(graph, predictions, labels,
    thresholds, options.from_logits == true, name, dtype)
  local sensitivity_values = true_positive_rate(graph, tp, fn,
    name .. "/sensitivity", dtype)
  local specificity = Ops.clean_div(graph, tn, tf.add(graph, tn, fp,
    name .. "/actual_negative", dtype), name .. "/specificity", dtype)
  return max_under_constraint(graph, sensitivity_values, specificity,
    sensitivity, name, dtype)
end

Metrics.SpecificityAtSensitivity = Metrics.specificity_at_sensitivity

-- Best precision whose operating point has recall >= the requested value,
-- following tf.keras.metrics.PrecisionAtRecall.
function Metrics.precision_at_recall(graph, predictions, labels, recall,
    options)
  if type(recall) == "table" and options == nil then
    options = recall
    recall = options.recall
  end
  options = options or {}
  if recall == nil then recall = options.recall end
  if not finite_number(recall) or recall < 0 or recall > 1 then
    error("precision_at_recall recall must be in [0, 1]", 4)
  end
  local name = options.name or "precision_at_recall"
  local dtype = options.dtype or "float32"
  local thresholds = sensitivity_thresholds(options.num_thresholds, name)
  local tp, fp, fn = thresholded_counts(graph, predictions, labels,
    thresholds, options.from_logits == true, name, dtype)
  local recall_values = true_positive_rate(graph, tp, fn,
    name .. "/recall", dtype)
  local precision = Ops.clean_div(graph, tp, tf.add(graph, tp, fp,
    name .. "/predicted_positive", dtype), name .. "/precision", dtype)
  return max_under_constraint(graph, recall_values, precision, recall, name,
    dtype)
end

Metrics.PrecisionAtRecall = Metrics.precision_at_recall

-- Best recall whose operating point has precision >= the requested value,
-- following tf.keras.metrics.RecallAtPrecision.
function Metrics.recall_at_precision(graph, predictions, labels, precision,
    options)
  if type(precision) == "table" and options == nil then
    options = precision
    precision = options.precision
  end
  options = options or {}
  if precision == nil then precision = options.precision end
  if not finite_number(precision) or precision < 0 or precision > 1 then
    error("recall_at_precision precision must be in [0, 1]", 4)
  end
  local name = options.name or "recall_at_precision"
  local dtype = options.dtype or "float32"
  local thresholds = sensitivity_thresholds(options.num_thresholds, name)
  local tp, fp, fn = thresholded_counts(graph, predictions, labels,
    thresholds, options.from_logits == true, name, dtype)
  local precision_values = Ops.clean_div(graph, tp, tf.add(graph, tp, fp,
    name .. "/predicted_positive", dtype), name .. "/precision", dtype)
  local recall = true_positive_rate(graph, tp, fn, name .. "/recall", dtype)
  return max_under_constraint(graph, precision_values, recall, precision, name,
    dtype)
end

Metrics.RecallAtPrecision = Metrics.recall_at_precision

function Metrics.fbeta_score(graph, predictions, labels, options)
  options = options or {}
  local name = options.name or "fbeta_score"
  local beta = options.beta or 2
  local threshold = options.threshold or 0.5
  local tp, fp, fn = binary_counts(graph, predictions, labels, threshold, name)
  local beta2 = tf.constant(graph, beta * beta, name .. "/beta2", "float32")
  local numerator = tf.mul(graph, tf.add(graph, tf.constant(graph, 1,
    name .. "/one_plus_beta", "float32"), beta2,
    name .. "/one_plus_beta2", "float32"), tp, name .. "/numerator",
    "float32")
  local denominator = tf.add(graph, numerator,
    tf.add(graph, tf.mul(graph, beta2, fp, name .. "/beta2_fp", "float32"), fn,
      name .. "/beta2_fp_fn", "float32"), name .. "/denominator", "float32")
  return Ops.clean_div(graph, numerator, denominator, name, "float32")
end

function Metrics.matthews_corrcoef(graph, predictions, labels, options)
  options = options or {}
  local name = options.name or "matthews_corrcoef"
  local tp, fp, fn, tn = binary_counts(graph, predictions, labels,
    options.threshold or 0.5, name)
  local numerator = tf.sub(graph, tf.mul(graph, tp, tn, name .. "/tp_tn",
    "float32"), tf.mul(graph, fp, fn, name .. "/fp_fn", "float32"),
    name .. "/numerator", "float32")
  local a = tf.add(graph, tp, fp, name .. "/tp_fp", "float32")
  local b = tf.add(graph, tp, fn, name .. "/tp_fn", "float32")
  local c = tf.add(graph, tn, fp, name .. "/tn_fp", "float32")
  local d = tf.add(graph, tn, fn, name .. "/tn_fn", "float32")
  local product = tf.mul(graph, tf.mul(graph, a, b, name .. "/ab", "float32"),
    tf.mul(graph, c, d, name .. "/cd", "float32"), name .. "/product",
    "float32")
  return Ops.clean_div(graph, numerator, tf.sqrt(graph, product,
    name .. "/denominator", "float32"), name, "float32")
end

function Metrics.r_square(graph, predictions, labels, options)
  options = options or {}
  local name = options.name or "r_square"
  local labels_float = tf.cast(graph, labels, labels.dtype or "float32",
    "float32", name .. "/labels")
  local mean = tf.reduce_mean(graph, labels_float, { 0 }, name .. "/mean",
    "float32", true)
  local residual = tf.sub(graph, labels_float, predictions,
    name .. "/residual", "float32")
  local total = tf.sub(graph, labels_float, mean, name .. "/total", "float32")
  local ss_res = Ops.sum_all(graph, tf.square(graph, residual,
    name .. "/squared_residual", "float32"), name .. "/ss_res", "float32")
  local ss_tot = Ops.sum_all(graph, tf.square(graph, total,
    name .. "/squared_total", "float32"), name .. "/ss_tot", "float32")
  return tf.sub(graph, tf.constant(graph, 1, name .. "/one", "float32"),
    Ops.clean_div(graph, ss_res, ss_tot, name .. "/ratio", "float32"),
    name, "float32")
end

local function is_graph_tensor(value)
  return type(value) == "table" and value.oper ~= nil
end

local function resolve_kappa_weights(options, name)
  local scheme = options.weights or options.weightage or "none"
  if scheme ~= "none" and scheme ~= "linear" and scheme ~= "quadratic" then
    error(name .. " weights must be 'none', 'linear', or 'quadratic'", 4)
  end
  return scheme
end

local function transpose_2d(graph, input, name)
  local perm = tf.constant(graph, { 1, 0 }, name .. "/perm", "int32")
  local desc = graph:new_operation("Transpose", name)
  desc:add_input(input)
  desc:add_input(perm)
  desc:set_type("T", "float32")
  desc:set_type("Tperm", "int32")
  return desc:finish()
end

local function class_range(graph, count, name)
  local start = tf.constant(graph, 0, name .. "/start", "int32")
  local delta = tf.constant(graph, 1, name .. "/delta", "int32")
  local desc = graph:new_operation("Range", name)
  desc:add_input(start)
  desc:add_input(count)
  desc:add_input(delta)
  desc:set_type("Tidx", "int32")
  return desc:finish()
end

local function identity_matrix(graph, count, name)
  local indices = class_range(graph, count, name .. "/indices")
  local row = tf.reshape(graph, indices, { -1, 1 }, name .. "/row", "int32")
  local column = tf.reshape(graph, indices, { 1, -1 }, name .. "/column",
    "int32")
  local desc = graph:new_operation("Equal", name .. "/equal")
  desc:add_input(row)
  desc:add_input(column)
  desc:set_type("T", "int32")
  return tf.cast(graph, desc:finish(), "bool", "float32", name .. "/matrix")
end

local function kappa_weights(graph, count, scheme, name)
  local indices = class_range(graph, count, name .. "/indices")
  local row = tf.reshape(graph, indices, { -1, 1 }, name .. "/row", "int32")
  local column = tf.reshape(graph, indices, { 1, -1 }, name .. "/column",
    "int32")
  if scheme == "none" then
    local desc = graph:new_operation("Equal", name .. "/equal")
    desc:add_input(row)
    desc:add_input(column)
    desc:set_type("T", "int32")
    return tf.cast(graph, desc:finish(), "bool", "float32", name .. "/weights")
  end
  local difference = tf.abs(graph, tf.sub(graph, row, column,
    name .. "/difference", "int32"), name .. "/absolute", "int32")
  local difference_float = tf.cast(graph, difference, "int32", "float32",
    name .. "/difference_float")
  local one = tf.constant(graph, 1, name .. "/one", "int32")
  local denominator = tf.cast(graph, tf.sub(graph, count, one,
    name .. "/classes_minus_one", "int32"), "int32", "float32",
    name .. "/denominator")
  local distance = tf.div(graph, difference_float, denominator,
    name .. "/distance", "float32")
  if scheme == "linear" then
    return tf.sub(graph, tf.constant(graph, 1, name .. "/one_float",
      "float32"), distance, name .. "/weights", "float32")
  end
  return tf.sub(graph, tf.constant(graph, 1, name .. "/one_float_q",
    "float32"), tf.mul(graph, distance, distance, name .. "/squared",
    "float32"), name .. "/weights", "float32")
end

local function weighted_kappa(graph, confusion, count, scheme, name)
  local total = Ops.sum_all(graph, confusion, name .. "/total", "float32")
  local weights = kappa_weights(graph, count, scheme, name .. "/weights")
  local observed = Ops.clean_div(graph, Ops.sum_all(graph, tf.mul(graph,
    confusion, weights, name .. "/weighted_confusion", "float32"),
    name .. "/observed_sum", "float32"), total, name .. "/observed", "float32")
  local actual = tf.reduce_sum(graph, confusion, { 1 }, name .. "/actual",
    "float32", false)
  local predicted = tf.reduce_sum(graph, confusion, { 0 },
    name .. "/predicted", "float32", false)
  local actual_matrix = tf.reshape(graph, actual, { -1, 1 },
    name .. "/actual_matrix", "float32")
  local predicted_matrix = tf.reshape(graph, predicted, { 1, -1 },
    name .. "/predicted_matrix", "float32")
  local expected_counts = tf.mul(graph, actual_matrix, predicted_matrix,
    name .. "/expected_counts", "float32")
  local expected = Ops.clean_div(graph, Ops.sum_all(graph, tf.mul(graph,
    expected_counts, weights, name .. "/expected_weighted", "float32"),
    name .. "/expected_sum", "float32"), tf.mul(graph, total, total,
    name .. "/total_square", "float32"), name .. "/expected", "float32")
  local one = tf.constant(graph, 1, name .. "/one", "float32")
  return Ops.clean_div(graph, tf.sub(graph, observed, expected,
    name .. "/difference", "float32"), tf.sub(graph, one, expected,
    name .. "/one_minus_expected", "float32"), name, "float32")
end

local function multiclass_confusion(graph, predictions, labels, options, name)
  local pred_classes, pred_one_hot = tensor_classes(graph, predictions,
    options, "predictions_one_hot", name .. "/predictions")
  local label_classes, label_one_hot = tensor_classes(graph, labels, options,
    "labels_one_hot", name .. "/labels")
  local count = class_count(graph, predictions, labels, pred_one_hot,
    label_one_hot, options, name .. "/classes")
  local pred_hot = one_hot_classes(graph, pred_classes, count,
    name .. "/predicted_one_hot")
  local label_hot = one_hot_classes(graph, label_classes, count,
    name .. "/labels_one_hot")
  local label_transposed = transpose_2d(graph, label_hot,
    name .. "/labels_transposed")
  return tf.matmul(graph, label_transposed, pred_hot, name .. "/matrix",
    "float32"), count
end

-- Mean intersection-over-union, matching tfa.metrics.MeanIoU.  Classes whose
-- union is zero are excluded from the mean; an all-empty batch returns zero.
function Metrics.mean_iou(graph, predictions, labels, num_classes, options)
  if type(num_classes) == "table" and options == nil then
    options = num_classes
    num_classes = options.num_classes
  end
  options = options or {}
  local name = options.name or "mean_iou"
  if num_classes ~= nil then
    validate_num_classes(num_classes, name)
    options.num_classes = num_classes
  end
  local pred_classes, pred_one_hot = tensor_classes(graph, predictions,
    options, "predictions_one_hot", name .. "/predictions")
  local label_classes, label_one_hot = tensor_classes(graph, labels, options,
    "labels_one_hot", name .. "/labels")
  local count = class_count(graph, predictions, labels, pred_one_hot,
    label_one_hot, options, name .. "/classes")
  local pred_hot = one_hot_classes(graph, pred_classes, count,
    name .. "/predicted_one_hot")
  local label_hot = one_hot_classes(graph, label_classes, count,
    name .. "/labels_one_hot")
  local confusion = tf.matmul(graph, transpose_2d(graph, label_hot,
    name .. "/labels_transposed"), pred_hot, name .. "/matrix", "float32")
  local diagonal = tf.reduce_sum(graph, tf.mul(graph, confusion,
    identity_matrix(graph, count, name .. "/identity"),
    name .. "/true_positive_tensor", "float32"), { 0 },
    name .. "/true_positive", "float32", false)
  local actual = tf.reduce_sum(graph, confusion, { 1 }, name .. "/actual",
    "float32", false)
  local predicted = tf.reduce_sum(graph, confusion, { 0 },
    name .. "/predicted", "float32", false)
  local union = tf.sub(graph, tf.add(graph, actual, predicted,
    name .. "/union_sum", "float32"), diagonal, name .. "/union", "float32")
  local valid = Ops.greater(graph, union, tf.constant(graph, 0,
    name .. "/zero", "float32"), name .. "/valid", "float32")
  local valid_count = Ops.sum_all(graph, tf.cast(graph, valid, "bool",
    "float32", name .. "/valid_float"), name .. "/valid_count", "float32")
  local iou = Ops.clean_div(graph, diagonal, union, name .. "/iou", "float32")
  return Ops.clean_div(graph, Ops.sum_all(graph, iou, name .. "/iou_sum",
    "float32"), valid_count, name, "float32")
end

Metrics.MeanIoU = Metrics.mean_iou

local function binary_matrix(graph, tensor, options, name)
  local dtype = tensor_dtype(tensor, options.dtype or "float32")
  local value = tensor
  if options.from_logits == true then
    value = tf.nn.sigmoid(graph, value, name .. "/sigmoid", dtype)
  end
  if options.threshold ~= nil then
    local threshold = tf.constant(graph, options.threshold,
      name .. "/threshold", dtype)
    local positive = Ops.greater(graph, value, threshold,
      name .. "/greater", dtype)
    return tf.cast(graph, positive, "bool", "float32", name .. "/binary")
  end
  return tf.cast(graph, value, dtype, "float32", name .. "/binary")
end

-- Multilabel confusion matrix, matching tfa.metrics.MultiLabelConfusionMatrix.
-- Returns a [num_classes, 2, 2] tensor whose rows are [true label] and columns
-- are [predicted label], so each class matrix is [[TN, FP], [FN, TP]].
function Metrics.multilabel_confusion_matrix(graph, predictions, labels,
    num_classes, options)
  if type(num_classes) == "table" and options == nil then
    options = num_classes
    num_classes = options.num_classes
  end
  options = options or {}
  local name = options.name or "multilabel_confusion_matrix"
  if num_classes == nil then num_classes = options.num_classes end
  validate_num_classes(num_classes, name)
  local predicted = binary_matrix(graph, predictions, options,
    name .. "/predictions")
  local truth = binary_matrix(graph, labels, options, name .. "/labels")
  local one = tf.constant(graph, 1, name .. "/one", "float32")
  local true_positive = tf.reduce_sum(graph, tf.mul(graph, truth, predicted,
    name .. "/true_positive_tensor", "float32"), { 0 },
    name .. "/true_positive", "float32", false)
  local predicted_sum = tf.reduce_sum(graph, predicted, { 0 },
    name .. "/predicted_sum", "float32", false)
  local truth_sum = tf.reduce_sum(graph, truth, { 0 },
    name .. "/truth_sum", "float32", false)
  local false_positive = tf.sub(graph, predicted_sum, true_positive,
    name .. "/false_positive", "float32")
  local false_negative = tf.sub(graph, truth_sum, true_positive,
    name .. "/false_negative", "float32")
  local truth_negative = tf.sub(graph, one, truth,
    name .. "/truth_negative", "float32")
  local predicted_negative = tf.sub(graph, one, predicted,
    name .. "/predicted_negative", "float32")
  local true_negative = tf.reduce_sum(graph, tf.mul(graph, truth_negative,
    predicted_negative, name .. "/true_negative_tensor", "float32"), { 0 },
    name .. "/true_negative", "float32", false)
  local components = { true_negative, false_positive, false_negative,
    true_positive }
  local columns = {}
  for index = 1, #components do
    columns[index] = tf.expand_dims(graph, components[index], 1,
      name .. "/component_" .. index, "float32")
  end
  local flat = tf.concat(graph, columns, 1, name .. "/flat", "float32")
  return tf.reshape(graph, flat, { num_classes, 2, 2 }, name, "float32")
end

Metrics.MultiLabelConfusionMatrix = Metrics.multilabel_confusion_matrix

function Metrics.cohen_kappa(graph, predictions, labels, options)
  if not is_graph_tensor(labels) then
    if options == nil then options = labels end
    labels = nil
  end
  options = options or {}
  local name = options.name or "cohen_kappa"
  validate_num_classes(options.num_classes, name)
  local scheme = resolve_kappa_weights(options, name)
  local confusion_matrix = labels == nil or options.confusion_matrix == true or
    options.is_confusion_matrix == true
  local multiclass = confusion_matrix or options.num_classes ~= nil or
    options.weights ~= nil or options.weightage ~= nil
  if multiclass then
    local confusion, count
    if confusion_matrix then
      confusion = tf.cast(graph, predictions, tensor_dtype(predictions),
        "float32", name .. "/confusion")
      if options.num_classes ~= nil then
        count = tf.constant(graph, options.num_classes, name .. "/classes",
          "int32")
      else
        count = last_dimension(graph, predictions, name .. "/classes")
      end
    else
      confusion, count = multiclass_confusion(graph, predictions, labels,
        options, name)
    end
    return weighted_kappa(graph, confusion, count, scheme, name)
  end
  local tp, fp, fn, tn = binary_counts(graph, predictions, labels,
    options.threshold or 0.5, name)
  local total = tf.add(graph, tf.add(graph, tp, fp, name .. "/total_pos",
    "float32"), tf.add(graph, fn, tn, name .. "/total_neg", "float32"),
    name .. "/total", "float32")
  local observed = Ops.clean_div(graph, tf.add(graph, tp, tn,
    name .. "/observed_correct", "float32"), total, name .. "/observed",
    "float32")
  local expected_positive = tf.mul(graph, tf.add(graph, tp, fp,
    name .. "/predicted_positive", "float32"), tf.add(graph, tp, fn,
    name .. "/actual_positive", "float32"), name .. "/expected_positive",
    "float32")
  local expected_negative = tf.mul(graph, tf.add(graph, tn, fp,
    name .. "/predicted_negative", "float32"), tf.add(graph, tn, fn,
    name .. "/actual_negative", "float32"), name .. "/expected_negative",
    "float32")
  local expected_total = tf.mul(graph, total, total, name .. "/total_square",
    "float32")
  local expected = Ops.clean_div(graph, tf.add(graph, expected_positive,
    expected_negative, name .. "/expected_sum", "float32"), expected_total,
    name .. "/expected", "float32")
  local one = tf.constant(graph, 1, name .. "/one_kappa", "float32")
  return Ops.clean_div(graph, tf.sub(graph, observed, expected,
    name .. "/difference", "float32"), tf.sub(graph, one, expected,
    name .. "/one_minus_expected", "float32"), name, "float32")
end

function Metrics.pearson_correlation(graph, predictions, labels, options)
  options = options or {}
  local name = options.name or "pearson_correlation"
  local labels_float = tf.cast(graph, labels, labels.dtype or "float32",
    "float32", name .. "/labels")
  local mean_prediction = tf.reduce_mean(graph, predictions, { 0 },
    name .. "/prediction_mean", "float32", true)
  local mean_label = tf.reduce_mean(graph, labels_float, { 0 },
    name .. "/label_mean", "float32", true)
  local centered_prediction = tf.sub(graph, predictions, mean_prediction,
    name .. "/prediction_centered", "float32")
  local centered_label = tf.sub(graph, labels_float, mean_label,
    name .. "/label_centered", "float32")
  local covariance = Ops.sum_all(graph, tf.mul(graph, centered_prediction,
    centered_label, name .. "/covariance_tensor", "float32"),
    name .. "/covariance", "float32")
  local pred_square = Ops.sum_all(graph, tf.square(graph, centered_prediction,
    name .. "/prediction_square", "float32"), name .. "/pred_square",
    "float32")
  local label_square = Ops.sum_all(graph, tf.square(graph, centered_label,
    name .. "/label_square", "float32"), name .. "/label_square_sum",
    "float32")
  local denominator = tf.sqrt(graph, tf.mul(graph, pred_square, label_square,
    name .. "/product", "float32"), name .. "/denominator", "float32")
  return Ops.clean_div(graph, covariance, denominator, name, "float32")
end

-- Geometric mean of all elements in a tensor, following
-- tfa.metrics.GeometricMean.
function Metrics.geometric_mean(graph, values, options)
  options = options or {}
  local name = options.name or "geometric_mean"
  local dtype = options.dtype or values.dtype or "float32"
  values = tf.cast(graph, values, values.dtype or dtype, dtype,
    name .. "/values")
  local logarithms = tf.log(graph, values, name .. "/log", dtype)
  local total = Ops.sum_all(graph, logarithms, name .. "/total", dtype)
  local ones = tf.add(graph, tf.mul(graph, values, tf.constant(graph, 0,
    name .. "/zero", dtype), name .. "/zero_values", dtype),
    tf.constant(graph, 1, name .. "/one", dtype), name .. "/ones", dtype)
  local count = Ops.sum_all(graph, ones, name .. "/count", dtype)
  return tf.exp(graph, tf.div(graph, total, count, name .. "/log_mean", dtype),
    name, dtype)
end

Metrics.GeometricMean = Metrics.geometric_mean

-- Harmonic mean of all elements in a tensor, following
-- tfa.metrics.HarmonicMean.
function Metrics.harmonic_mean(graph, values, options)
  options = options or {}
  local name = options.name or "harmonic_mean"
  local dtype = options.dtype or values.dtype or "float32"
  values = tf.cast(graph, values, values.dtype or dtype, dtype,
    name .. "/values")
  local reciprocal = tf.div(graph, tf.constant(graph, 1, name .. "/one",
    dtype), values, name .. "/reciprocal", dtype)
  local total = Ops.sum_all(graph, reciprocal, name .. "/total", dtype)
  local ones = tf.add(graph, tf.mul(graph, values, tf.constant(graph, 0,
    name .. "/zero", dtype), name .. "/zero_values", dtype),
    tf.constant(graph, 1, name .. "/ones_one", dtype), name .. "/ones", dtype)
  local count = Ops.sum_all(graph, ones, name .. "/count", dtype)
  return Ops.clean_div(graph, count, total, name, dtype)
end

Metrics.HarmonicMean = Metrics.harmonic_mean

-- Macro F1 for dense one-hot predictions/labels.  Class predictions are
-- compared one-vs-rest and averaged uniformly.
function Metrics.multiclass_f1(graph, predictions, labels, num_classes, options)
  options = options or {}
  local name = options.name or "multiclass_f1"
  if type(num_classes) ~= "number" or num_classes < 2 then
    error("multiclass_f1 requires num_classes >= 2", 3)
  end
  local dtype = "float32"
  local pred_classes = tf.argmax(graph, predictions, 1, name .. "/pred",
    dtype, "int64")
  local label_classes
  if labels.dtype == "int32" or labels.dtype == "int64" then
    label_classes = tf.cast(graph, labels, labels.dtype, "int64",
      name .. "/labels")
  else
    label_classes = tf.argmax(graph, labels, 1, name .. "/label", dtype,
      "int64")
  end
  pred_classes = tf.cast(graph, pred_classes, "int64", "int64",
    name .. "/pred64")
  label_classes = tf.cast(graph, label_classes, "int64", "int64",
    name .. "/label64")
  local epsilon = tf.constant(graph, 1e-8, name .. "/epsilon", dtype)
  local pred_hot = tf.one_hot(graph, pred_classes, num_classes,
    { dtype = dtype, indices_dtype = "int64", name = name .. "/pred_hot" })
  local label_hot = tf.one_hot(graph, label_classes, num_classes,
    { dtype = dtype, indices_dtype = "int64", name = name .. "/label_hot" })
  local total
  for class = 0, num_classes - 1 do
    local axis = tf.one_hot(graph, tf.constant(graph, class,
      name .. "/class_" .. class, "int64"), num_classes,
      { dtype = dtype, indices_dtype = "int64",
        name = name .. "/axis_" .. class })
    local pred_c = tf.reduce_sum(graph, tf.mul(graph, pred_hot, axis,
      name .. "/pred_one_" .. class, dtype), { 0, 1 },
      name .. "/pred_count_" .. class, dtype, false)
    local label_c = tf.reduce_sum(graph, tf.mul(graph, label_hot, axis,
      name .. "/label_one_" .. class, dtype), { 0, 1 },
      name .. "/label_count_" .. class, dtype, false)
    local tp = tf.mul(graph, pred_c, label_c, name .. "/tp_" .. class, dtype)
    local denominator = tf.add(graph, tf.add(graph, pred_c, label_c,
      name .. "/sum_" .. class, dtype), epsilon, name .. "/denom_" .. class,
      dtype)
    local f1 = Ops.clean_div(graph, tf.mul(graph, tf.constant(graph, 2,
      name .. "/two_" .. class, dtype), tp, name .. "/num_" .. class, dtype),
      denominator, name .. "/f1_" .. class, dtype)
    total = total == nil and f1 or tf.add(graph, total, f1,
      name .. "/total_" .. class, dtype)
  end
  return tf.div(graph, total, tf.constant(graph, num_classes,
    name .. "/count", dtype), name, dtype)

end


-- Hamming distance between two tensors, matching tfa.metrics.hamming_distance.
-- The result is the fraction of element positions whose values differ.
function Metrics.hamming_distance(graph, actuals, predictions, options)
  options = options or {}
  local name = options.name or "hamming_distance"
  local dtype = predictions.dtype or actuals.dtype or "float32"
  actuals = tf.cast(graph, actuals, actuals.dtype or dtype, dtype,
    name .. "/actuals")
  predictions = tf.cast(graph, predictions, predictions.dtype or dtype, dtype,
    name .. "/predictions")
  local mismatch = Ops.not_equal(graph, actuals, predictions,
    name .. "/mismatch", dtype)
  local values = tf.cast(graph, mismatch, "bool", "float32",
    name .. "/mismatch_float")
  return tf.reduce_mean(graph, Ops.flatten(graph, values, name .. "/flat",
    "float32"), { 0 }, name, "float32")
end

-- Hamming loss, matching tfa.metrics.HammingLoss.  In "multiclass" mode each
-- row is expected to contain one label and the loss is one minus the fraction
-- of rows whose active positions match.  "multilabel" mode counts per-label
-- mismatches.  A nil threshold selects the per-row maximum (with zero rows
-- treated as inactive), like TensorFlow Addons.
function Metrics.hamming_loss(graph, predictions, labels, options)
  options = options or {}
  local name = options.name or "hamming_loss"
  local mode = options.mode or "multiclass"
  if mode ~= "multiclass" and mode ~= "multilabel" then
    error("hamming_loss mode must be 'multiclass' or 'multilabel'", 4)
  end
  local dtype = options.dtype or "float32"
  local truth = tf.cast(graph, labels, labels.dtype or "int32", "int32",
    name .. "/labels")
  local scores = predictions
  local score_dtype = predictions.dtype or dtype
  local predicted
  local threshold = options.threshold
  if threshold == nil then
    local maximum = tf.reduce_max(graph, scores, { 1 }, name .. "/maximum",
      score_dtype, true)
    local at_maximum = Ops.greater_equal(graph, scores, maximum,
      name .. "/at_maximum", score_dtype)
    local absolute = tf.abs(graph, scores, name .. "/absolute", score_dtype)
    local epsilon = tf.constant(graph, 1e-12, name .. "/epsilon", score_dtype)
    local nonzero = Ops.greater(graph, absolute, epsilon, name .. "/nonzero",
      score_dtype)
    predicted = Ops.logical_and(graph, at_maximum, nonzero,
      name .. "/predicted")
  else
    if type(threshold) ~= "number" or threshold ~= threshold or
        threshold == math.huge or threshold == -math.huge then
      error("hamming_loss threshold must be a finite number", 4)
    end
    predicted = Ops.greater(graph, scores, tf.constant(graph, threshold,
      name .. "/threshold", score_dtype), name .. "/predicted", score_dtype)
  end
  local one = tf.constant(graph, 1, name .. "/one", dtype)
  if mode == "multiclass" then
    local zero = tf.constant(graph, 0, name .. "/zero_int", "int32")
    local active = Ops.not_equal(graph, truth, zero, name .. "/active",
      "int32")
    local matched = Ops.logical_and(graph, active, predicted,
      name .. "/matched")
    local matched_float = tf.cast(graph, matched, "bool", dtype,
      name .. "/matched_float")
    local counts = tf.reduce_sum(graph, matched_float, { 1 },
      name .. "/counts", dtype, false)
    local losses = tf.sub(graph, one, counts, name .. "/losses", dtype)
    return tf.reduce_mean(graph, losses, { 0 }, name, dtype)
  end
  local predicted_int = tf.cast(graph, predicted, "bool", "int32",
    name .. "/predicted_int")
  local mismatch = Ops.not_equal(graph, truth, predicted_int,
    name .. "/mismatch", "int32")
  local mismatch_float = tf.cast(graph, mismatch, "bool", dtype,
    name .. "/mismatch_float")
  local counts = tf.reduce_sum(graph, mismatch_float, { 1 },
    name .. "/mismatch_counts", dtype, false)
  local classes
  if options.num_classes ~= nil then
    validate_num_classes(options.num_classes, name)
    classes = tf.constant(graph, options.num_classes, name .. "/classes",
      "int32")
  else
    classes = last_dimension(graph, truth, name .. "/classes", "int32")
  end
  local denominator = tf.cast(graph, classes, "int32", dtype,
    name .. "/denominator")
  local losses = tf.div(graph, counts, denominator, name .. "/losses", dtype)
  return tf.reduce_mean(graph, losses, { 0 }, name, dtype)
end

Metrics.HammingLoss = Metrics.hamming_loss

return Metrics
