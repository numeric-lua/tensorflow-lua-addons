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

function Metrics.f1_score(graph, predictions, labels, options)
  options = options or {}
  local name = options.name or "f1_score"
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

function Metrics.cohen_kappa(graph, predictions, labels, options)
  options = options or {}
  local name = options.name or "cohen_kappa"
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

return Metrics
