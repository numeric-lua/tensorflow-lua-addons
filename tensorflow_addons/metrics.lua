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

local function last_dimension(graph, tensor, name)
  local shape = tf.shape(graph, tensor, name .. "/input_shape",
    tensor_dtype(tensor))
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

local function multiclass_f1(graph, predictions, labels, options, name)
  local average = options.average or "macro"
  if average ~= "macro" and average ~= "micro" and average ~= "weighted" then
    error("f1_score average must be 'macro', 'micro', or 'weighted'", 4)
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
  local two = tf.constant(graph, 2, name .. "/two", "float32")
  local per_class = Ops.clean_div(graph, tf.mul(graph, two, true_positive,
    name .. "/numerator", "float32"), tf.add(graph, tf.mul(graph, two,
    true_positive, name .. "/two_tp", "float32"), tf.add(graph,
    false_positive, false_negative, name .. "/errors", "float32"),
    name .. "/denominator", "float32"), name .. "/per_class", "float32")
  if average == "macro" then
    return tf.reduce_mean(graph, per_class, { 0 }, name, "float32", false)
  end
  if average == "micro" then
    local tp = Ops.sum_all(graph, true_positive, name .. "/tp_total",
      "float32")
    local fp = Ops.sum_all(graph, false_positive, name .. "/fp_total",
      "float32")
    local fn = Ops.sum_all(graph, false_negative, name .. "/fn_total",
      "float32")
    local numerator = tf.mul(graph, two, tp, name .. "/micro_numerator",
      "float32")
    return Ops.clean_div(graph, numerator, tf.add(graph, numerator,
      tf.add(graph, fp, fn, name .. "/micro_errors", "float32"),
      name .. "/micro_denominator", "float32"), name, "float32")
  end
  local weighted = tf.mul(graph, per_class, label_support,
    name .. "/weighted", "float32")
  return Ops.clean_div(graph, Ops.sum_all(graph, weighted,
    name .. "/weighted_sum", "float32"), Ops.sum_all(graph, label_support,
    name .. "/support_sum", "float32"), name, "float32")
end

function Metrics.f1_score(graph, predictions, labels, options)
  options = options or {}
  local name = options.name or "f1_score"
  validate_num_classes(options.num_classes, name)
  local multiclass = options.average ~= nil or options.num_classes ~= nil or
    options.predictions_one_hot ~= nil or options.labels_one_hot ~= nil
  if multiclass then
    return multiclass_f1(graph, predictions, labels, options, name)
  end
  local threshold = options.threshold or 0.5
  local tp, fp, fn = binary_counts(graph, predictions, labels, threshold, name)
  local numerator = tf.mul(graph, tf.constant(graph, 2, name .. "/two",
    "float32"), tp, name .. "/numerator", "float32")
  local denominator = tf.add(graph, numerator, tf.add(graph, fp, fn,
    name .. "/errors", "float32"), name .. "/denominator", "float32")
  return Ops.clean_div(graph, numerator, denominator, name, "float32")
end

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


return Metrics
