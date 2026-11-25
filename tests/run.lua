local root = (arg and arg[0] or "tests/run.lua"):match("^(.*)tests/run.lua$")
  or "."
local tf_root = os.getenv("TENSORFLOW_LUA_ROOT")
if not tf_root or not io.open(tf_root .. "/tensorflow/init.lua", "r") then
  tf_root = root .. "/../tensorflow-lua"
  if not io.open(tf_root .. "/tensorflow/init.lua", "r") then
    tf_root = root .. "/tensorflow-lua"
  end
end
package.path = root .. "/?.lua;" .. root .. "/?/init.lua;" ..
  tf_root .. "/?.lua;" .. tf_root .. "/?/init.lua;" .. package.path

local tf = require("tensorflow")
local tfa = require("tensorflow_addons")
local passed, failed = 0, 0
local failures = {}

local function test(name, fn)
  local ok, err = pcall(fn)
  if ok then
    passed = passed + 1
    io.write(".")
  else
    failed = failed + 1
    failures[#failures + 1] = name .. "\n  " .. tostring(err)
    io.write("F")
  end
  io.flush()
end

local function assert_eq(actual, expected, label)
  if actual ~= expected then
    error(string.format("%s: expected %s, got %s", label or "value",
      tostring(expected), tostring(actual)), 2)
  end
end

local function assert_true(value, label)
  if not value then error(label or "expected true", 2) end
end

local function near(actual, expected, tolerance, label)
  if math.abs(actual - expected) > (tolerance or 1e-5) then
    error(string.format("%s: expected %s, got %s", label or "value",
      tostring(expected), tostring(actual)), 2)
  end
end

local function run(graph, feeds, output)
  local session = tf.Session.new(graph)
  local value = session:run(feeds, { output })[1]:totable()
  session:close()
  return value
end

test("activation outputs are finite and shaped", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 4 }, "x")
  local activations = {
    tfa.activations.gelu(graph, input, "gelu"),
    tfa.activations.swish(graph, input, "swish"),
    tfa.activations.mish(graph, input, "mish"),
    tfa.activations.lisht(graph, input, "lisht"),
    tfa.activations.tanhshrink(graph, input, "tanhshrink"),
  }
  local session = tf.Session.new(graph)
  local values = session:run({ [input] = { -1, 0, 1, 2 } }, activations)
  for index, tensor in ipairs(values) do
    assert_eq(#tensor:shape(), 1)
    for _, value in ipairs(tensor:totable()) do
      assert_true(value == value, "activation " .. index .. " produced NaN")
    end
  end
  near(values[2]:totable()[2], 0, 1e-6, "swish(0)")
  near(values[3]:totable()[2], 0, 1e-6, "mish(0)")
  session:close()
end)

test("focal loss matches the binary formula", function()
  local graph = tf.Graph.new()
  local labels = tf.placeholder(graph, "float32", { 2, 2 }, "labels")
  local predictions = tf.placeholder(graph, "float32", { 2, 2 },
    "predictions")
  local loss = tfa.losses.focal(graph, labels, predictions,
    { gamma = 2, alpha = 0.25 })
  local value = run(graph, { [labels] = { { 1, 0 }, { 1, 0 } },
    [predictions] = { { 0.9, 0.1 }, { 0.9, 0.1 } } }, loss)
  near(value,
    (-0.25 * 0.01 * math.log(0.9) - 0.75 * 0.01 * math.log(0.9)) / 2,
    1e-6, "focal")
end)

test("dice and tversky losses vanish for perfect overlap", function()
  local graph = tf.Graph.new()
  local labels = tf.placeholder(graph, "float32", { 2, 2 }, "labels")
  local predictions = tf.placeholder(graph, "float32", { 2, 2 },
    "predictions")
  local dice = tfa.losses.dice(graph, labels, predictions)
  local tversky = tfa.losses.tversky(graph, labels, predictions)
  local session = tf.Session.new(graph)
  local values = session:run({
    [labels] = { { 1, 0 }, { 0, 1 } },
    [predictions] = { { 1, 0 }, { 0, 1 } },
  }, { dice, tversky })
  near(values[1]:totable(), 0, 1e-6, "dice")
  near(values[2]:totable(), 0, 1e-6, "tversky")
  session:close()
end)

test("binary metrics are exact for perfect predictions", function()
  local graph = tf.Graph.new()
  local labels = tf.placeholder(graph, "float32", { 4 }, "labels")
  local predictions = tf.placeholder(graph, "float32", { 4 }, "predictions")
  local outputs = {
    tfa.metrics.f1_score(graph, predictions, labels),
    tfa.metrics.fbeta_score(graph, predictions, labels, { beta = 2 }),
    tfa.metrics.matthews_corrcoef(graph, predictions, labels),
    tfa.metrics.r_square(graph, predictions, labels),
    tfa.metrics.pearson_correlation(graph, predictions, labels),
  }
  local session = tf.Session.new(graph)
  local values = session:run({
    [labels] = { 1, 0, 1, 0 },
    [predictions] = { 1, 0, 1, 0 },
  }, outputs)
  near(values[1]:totable(), 1, 1e-6, "f1")
  near(values[2]:totable(), 1, 1e-6, "fbeta")
  near(values[3]:totable(), 1, 1e-6, "matthews")
  near(values[4]:totable(), 1, 1e-6, "r2")
  near(values[5]:totable(), 1, 1e-6, "pearson")
  session:close()
end)

test("group normalization produces zero-mean groups", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 2, 4 }, "x")
  local layer = tfa.layers.group_normalization(graph, input, 2,
    { input_shape = { 4 }, name = "groupnorm" })
  local session = tf.Session.new(graph)
  session:run({}, {}, layer.initializers)
  local output = session:run({ [input] = { { 1, 2, 3, 4 },
    { 2, 3, 4, 5 } } }, { layer.output })[1]:totable()
  near((output[1][1] + output[1][2] + output[2][1] + output[2][2]) / 4, 0,
    1e-5, "first group")
  near((output[1][3] + output[1][4] + output[2][3] + output[2][4]) / 4, 0,
    1e-5, "second group")
  session:close()
end)

test("layer and instance normalization preserve shapes", function()
  local graph = tf.Graph.new()
  local flat = tf.placeholder(graph, "float32", { 2, 4 }, "flat")
  local image = tf.placeholder(graph, "float32", { 1, 2, 2, 4 }, "image")
  local layer = tfa.layers.layer_normalization(graph, flat,
    { input_shape = { 4 }, name = "layernorm" })
  local instance = tfa.layers.instance_normalization(graph, image,
    { input_shape = { 2, 2, 4 }, name = "instancenorm" })
  local session = tf.Session.new(graph)
  session:run({}, {}, layer.initializers)
  session:run({}, {}, instance.initializers)
  local flat_output = session:run({ [flat] = { { 1, 2, 3, 4 },
    { 2, 3, 4, 5 } } }, { layer.output })[1]
  local image_output = session:run({ [image] = { { { { 1, 2, 3, 4 },
    { 2, 3, 4, 5 } }, { { 3, 4, 5, 6 }, { 4, 5, 6, 7 } } } } },
    { instance.output })[1]
  assert_eq(table.concat(flat_output:shape(), ","), "2,4")
  assert_eq(table.concat(image_output:shape(), ","), "1,2,2,4")
  session:close()
end)

test("callbacks stop on non-finite values", function()
  local callback = tfa.callbacks.TerminateOnNaN()
  local history = { loss = { 1, 2, 0 / 0 } }
  callback.on_epoch_end(history)
  assert_eq(history.stop_training, true)
  assert_eq(callback.stopped_by, "loss")
end)

test("BackupAndRestore delegates to the model", function()
  local calls = {}
  local model = {
    load_training = function(_, path) calls.load = path; return true end,
    save_training = function(_, path) calls.save = path end,
  }
  local callback = tfa.callbacks.BackupAndRestore("/tmp/addons-backup")
  callback.on_train_begin({}, model)
  callback.on_epoch_end({}, 1, model)
  assert_eq(calls.load, "/tmp/addons-backup")
  assert_eq(calls.save, "/tmp/addons-backup")
  assert_eq(callback.restored, true)
  assert_eq(callback.saved, true)
end)

test("LAMB updates variables and optimizer state", function()
  math.randomseed(46)
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 2, 1 }, "input")
  local target = tf.placeholder(graph, "float32", { 2, 1 }, "target")
  local weight = require("tensorflow.train").variable(graph, { { 0.5 } },
    "weight", { dtype = "float32" })
  local output = tf.matmul(graph, input, weight.read, "output", "float32")
  local loss = tf.losses.mean_squared_error(graph, output, target,
    { axes = { 0, 1 } })
  local targets, _, state = tfa.optimizers.LAMB(graph, loss, { weight },
    { learning_rate = 0.01 })
  local session = tf.Session.new(graph)
  session:run({}, {}, { weight.initializer, unpack(state.initializers) })
  session:run({ [input] = { { 1 }, { 2 } }, [target] = { { 1 }, { 2 } } }, {},
    targets)
  local value = session:run({}, { weight.read })[1]:totable()[1][1]
  assert_true(value == value, "LAMB produced NaN")
  assert_true(value ~= 0.5, "LAMB did not update the weight")
  session:close()
end)


test("Cohen's kappa is one for perfect predictions", function()
  local graph = tf.Graph.new()
  local labels = tf.placeholder(graph, "float32", { 4 }, "labels")
  local predictions = tf.placeholder(graph, "float32", { 4 }, "predictions")
  local metric = tfa.metrics.cohen_kappa(graph, predictions, labels)
  local value = run(graph, { [labels] = { 1, 0, 1, 0 },
    [predictions] = { 1, 0, 1, 0 } }, metric)
  near(value, 1, 1e-6, "kappa")
end)

test("CSVLogger writes a header and epochs", function()
  local path = "/tmp/tfa-csv-logger.csv"
  os.remove(path)
  local callback = tfa.callbacks.CSVLogger(path)
  callback.on_train_begin({ loss = {} })
  callback.on_epoch_end({ loss = { 1.5 } })
  callback.on_epoch_end({ loss = { 0.5 } })
  local file = assert(io.open(path, "r"))
  local content = file:read("*a")
  file:close()
  os.remove(path)
  assert_true(content:find("loss", 1, true) ~= nil)
  assert_true(content:find("1.5", 1, true) ~= nil)
  assert_true(content:find("0.5", 1, true) ~= nil)
end)

test("NovoGrad updates variables and optimizer state", function()
  math.randomseed(47)
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 2, 1 }, "ng_input")
  local target = tf.placeholder(graph, "float32", { 2, 1 }, "ng_target")
  local weight = require("tensorflow.train").variable(graph, { { 0.5 } },
    "ng_weight", { dtype = "float32" })
  local output = tf.matmul(graph, input, weight.read, "ng_output",
    "float32")
  local loss = tf.losses.mean_squared_error(graph, output, target,
    { axes = { 0, 1 } })
  local targets, _, state = tfa.optimizers.NovoGrad(graph, loss, { weight },
    { learning_rate = 0.01 })
  local session = tf.Session.new(graph)
  session:run({}, {}, { weight.initializer, unpack(state.initializers) })
  session:run({ [input] = { { 1 }, { 2 } }, [target] = { { 1 }, { 2 } } }, {},
    targets)
  local value = session:run({}, { weight.read })[1]:totable()[1][1]
  assert_true(value == value, "NovoGrad produced NaN")
  assert_true(value ~= 0.5, "NovoGrad did not update the weight")
  session:close()
end)


test("Lookahead rejects invalid sync periods", function()
  for _, period in ipairs({ 0, -2, 1.5, "5" }) do
    local ok, err = pcall(tfa.optimizers.Lookahead, tfa.optimizers.LAMB,
      { sync_period = period })
    assert_true(not ok, "Lookahead should reject period " .. tostring(period))
    assert_true(tostring(err):find("sync_period", 1, true) ~= nil,
      "invalid period error should mention sync_period")
  end
end)

test("Lookahead leaves slow weights alone until the sync step", function()
  local graph = tf.Graph.new()
  local weight = require("tensorflow.train").variable(graph, { { 1.0 } },
    "la_weight", { dtype = "float32" })
  local function base(g, _, variables)
    local variable = variables[1]
    local delta = tf.constant(g, { { 0.1 } }, "la_base/delta", "float32")
    local next_weight = tf.sub(g, variable.read, delta, "la_base/next",
      "float32")
    local desc = g:new_operation("Assign", "la_base/assign")
    desc:add_input(variable.ref)
    desc:add_input(next_weight)
    desc:set_type("T", "float32")
    desc:set_bool("use_locking", true)
    return { desc:finish() }
  end
  local optimizer = tfa.optimizers.Lookahead(base,
    { sync_period = 3, slow_step_size = 0.5 })
  local targets, gradients, state = optimizer(graph, nil, { weight })
  assert_eq(#gradients, 0)
  local slow = state.slow_weights[1]
  local session = tf.Session.new(graph)
  session:run({}, {}, { weight.initializer, unpack(state.initializers) })
  local function read(variable) return session:run({}, { variable.read })[1]
    :totable()[1][1] end

  local desc = graph:new_operation("Assign", "la_set_slow")
  desc:add_input(slow.ref)
  desc:add_input(tf.constant(graph, { { 10.0 } }, "la_set_slow/value",
    "float32"))
  desc:set_type("T", "float32")
  desc:set_bool("use_locking", true)
  session:run({}, {}, { desc:finish() })
  assert_eq(read(slow), 10)

  session:run({}, {}, targets)
  near(read(weight), 0.9, 1e-6, "fast weight after first step")
  assert_eq(read(slow), 10)
  session:run({}, {}, targets)
  near(read(weight), 0.8, 1e-6, "fast weight after second step")
  assert_eq(read(slow), 10)

  local before_sync = read(weight)
  session:run({}, {}, targets)
  local fast = read(weight)
  local synced = read(slow)
  near(fast, synced, 1e-6, "fast and slow weights after sync")
  near(synced, 5.35, 1e-5, "blended slow weight")
  assert_true(fast > before_sync, "fast weight should move toward slow")
  assert_true(synced < 10, "slow weight should move toward fast")
  near(session:run({}, { state.step.read })[1]:totable(), 3, 0, "step")
  session:close()
end)

test("Lookahead defaults initialize slow weights from fast weights", function()
  local graph = tf.Graph.new()
  local weight = require("tensorflow.train").variable(graph, { { 3.0 } },
    "la_default_weight", { dtype = "float32" })
  local function base(g, _, variables)
    local variable = variables[1]
    local delta = tf.constant(g, { { 0.1 } }, "la_default/delta", "float32")
    local next_weight = tf.sub(g, variable.read, delta, "la_default/next",
      "float32")
    local desc = g:new_operation("Assign", "la_default/assign")
    desc:add_input(variable.ref)
    desc:add_input(next_weight)
    desc:set_type("T", "float32")
    desc:set_bool("use_locking", true)
    return { desc:finish() }
  end
  local optimizer = tfa.optimizers.Lookahead(base)
  local targets, _, state = optimizer(graph, nil, { weight })
  assert_eq(state.sync_period, 5)
  assert_eq(state.slow_step_size, 0.5)
  local session = tf.Session.new(graph)
  session:run({}, {}, { unpack(state.initializers) })
  local function read(variable) return session:run({}, { variable.read })[1]
    :totable()[1][1] end
  near(read(weight), 3, 1e-6, "initialized fast weight")
  near(read(state.slow_weights[1]), 3, 1e-6, "initialized slow weight")
  session:run({}, {}, targets)
  near(read(weight), 2.9, 1e-6, "fast weight after non-sync step")
  near(read(state.slow_weights[1]), 3, 1e-6, "slow weight before sync")
  session:close()
end)

test("Lookahead wraps LAMB and synchronizes after the period", function()
  math.randomseed(48)
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 2, 1 }, "la_input")
  local target = tf.placeholder(graph, "float32", { 2, 1 }, "la_target")
  local weight = require("tensorflow.train").variable(graph, { { 2.0 } },
    "la_lamb_weight", { dtype = "float32" })
  local output = tf.matmul(graph, input, weight.read, "la_output", "float32")
  local loss = tf.losses.mean_squared_error(graph, output, target,
    { axes = { 0, 1 } })
  local optimizer = tfa.optimizers.Lookahead(tfa.optimizers.LAMB,
    { sync_period = 2, slow_step_size = 0.5, learning_rate = 0.05 })
  local targets, _, state = optimizer(graph, loss, { weight })
  local session = tf.Session.new(graph)
  session:run({}, {}, { weight.initializer, unpack(state.initializers) })
  local function read(variable) return session:run({}, { variable.read })[1]
    :totable()[1][1] end
  local feeds = { [input] = { { 1 }, { 2 } }, [target] = { { 1 }, { 2 } } }
  session:run(feeds, {}, targets)
  near(read(state.slow_weights[1]), 2, 1e-6, "slow weight after first step")
  assert_true(read(weight) ~= 2, "LAMB did not update the fast weight")
  session:run(feeds, {}, targets)
  near(read(weight), read(state.slow_weights[1]), 1e-6,
    "LAMB fast and slow weights after sync")
  session:close()
end)

test("triplet loss uses the configured margin", function()
  local graph = tf.Graph.new()
  local anchor = tf.placeholder(graph, "float32", { 1 }, "anchor")
  local positive = tf.placeholder(graph, "float32", { 1 }, "positive")
  local negative = tf.placeholder(graph, "float32", { 1 }, "negative")
  local loss = tfa.losses.triplet(graph, anchor, positive, negative,
    { margin = 1 })
  local session = tf.Session.new(graph)
  local zero = session:run({ [anchor] = { 0 }, [positive] = { 0 },
    [negative] = { 1 } }, { loss })[1]:totable()
  local one = session:run({ [anchor] = { 0 }, [positive] = { 1 },
    [negative] = { 1 } }, { loss })[1]:totable()
  near(zero, 0, 1e-6, "triplet zero")
  near(one, 1, 1e-6, "triplet margin")
  session:close()
end)

test("weight normalization has unit column norms", function()
  math.randomseed(49)
  local graph = tf.Graph.new()
  local layer = tfa.layers.weight_normalization(graph, { 3, 2 },
    { name = "wn" })
  local session = tf.Session.new(graph)
  session:run({}, {}, layer.initializers)
  local values = session:run({}, { layer.output })[1]:totable()
  for column = 1, 2 do
    local total = 0
    for row = 1, 3 do total = total + values[row][column] ^ 2 end
    near(total, 1, 1e-5, "column " .. column)
  end
  session:close()
end)


test("contrastive loss grows with labeled distances", function()
  local graph = tf.Graph.new()
  local labels = tf.placeholder(graph, "float32", { 2 }, "labels")
  local left = tf.placeholder(graph, "float32", { 2, 2 }, "left")
  local right = tf.placeholder(graph, "float32", { 2, 2 }, "right")
  local loss = tfa.losses.contrastive(graph, labels, left, right,
    { margin = 1 })
  local session = tf.Session.new(graph)
  local close = session:run({ [labels] = { 1, 0 }, [left] = { { 0, 0 }, { 0, 0 } },
    [right] = { { 0, 0 }, { 2, 0 } } }, { loss })[1]:totable()
  local far = session:run({ [labels] = { 1, 0 }, [left] = { { 0, 0 }, { 0, 0 } },
    [right] = { { 2, 0 }, { 2, 0 } } }, { loss })[1]:totable()
  assert_true(far > close, "contrastive loss should increase")
  session:close()
end)

test("giou loss matches analytic values for paired boxes", function()
  local cases = {
    { key = "identical", truth = { 0, 0, 2, 2 },
      predicted = { 0, 0, 2, 2 }, expected = 0 },
    { key = "disjoint", truth = { 0, 0, 2, 2 },
      predicted = { 3, 0, 5, 2 }, expected = 6 / 5 },
    { key = "partial", truth = { 0, 0, 2, 2 },
      predicted = { 1, 1, 3, 3 }, expected = 68 / 63 },
    { key = "nested", truth = { 0, 0, 4, 4 },
      predicted = { 1, 1, 3, 3 }, expected = 3 / 4 },
  }
  for _, case in ipairs(cases) do
    local graph = tf.Graph.new()
    local labels = tf.placeholder(graph, "float32", { 1, 4 },
      case.key .. "_labels")
    local predictions = tf.placeholder(graph, "float32", { 1, 4 },
      case.key .. "_predictions")
    local loss = tfa.losses.giou(graph, labels, predictions)
    local value = run(graph, { [labels] = { case.truth },
      [predictions] = { case.predicted } }, loss)
    near(value, case.expected, 1e-6, "giou " .. case.key)
  end
end)

test("giou loss averages one-vs-one batch pairs", function()
  local graph = tf.Graph.new()
  local labels = tf.placeholder(graph, "float32", { 2, 4 }, "giou_batch_labels")
  local predictions = tf.placeholder(graph, "float32", { 2, 4 },
    "giou_batch_predictions")
  local loss = tfa.losses.giou(graph, labels, predictions)
  local value = run(graph, {
    [labels] = { { 0, 0, 2, 2 }, { 0, 0, 4, 4 } },
    [predictions] = { { 0, 0, 2, 2 }, { 1, 1, 3, 3 } },
  }, loss)
  near(value, 3 / 8, 1e-6, "giou batch mean")
end)

test("giou loss is finite for degenerate boxes", function()
  local graph = tf.Graph.new()
  local labels = tf.placeholder(graph, "float32", { 2, 4 },
    "giou_degenerate_labels")
  local predictions = tf.placeholder(graph, "float32", { 2, 4 },
    "giou_degenerate_predictions")
  local loss = tfa.losses.giou(graph, labels, predictions)
  local value = run(graph, {
    [labels] = { { 0, 0, 0, 0 }, { 1, 1, 1, 1 } },
    [predictions] = { { 0, 0, 0, 0 }, { 2, 2, 2, 2 } },
  }, loss)
  assert_true(value == value, "giou degenerate boxes produced NaN")
  assert_true(value >= 0, "giou degenerate boxes produced a negative loss")
end)

test("spectral normalization has a finite norm and updates state", function()
  local graph = tf.Graph.new()
  local weight = tf.variable(graph, { { 2, 0 }, { 0, 1 } }, "spectral_weight",
    { dtype = "float32" })
  local layer = tfa.layers.spectral_normalization(graph, weight.read,
    { weight_shape = { 2, 2 }, iterations = 3, name = "spectral" })
  local session = tf.Session.new(graph)
  session:run({}, {}, { weight.initializer, unpack(layer.initializers) })
  local values = session:run({}, { layer.output })[1]:totable()
  local largest = 0
  for row = 1, 2 do
    for column = 1, 2 do
      largest = math.max(largest, math.abs(values[row][column]))
    end
  end
  assert_true(largest <= 1.01, "spectral weight should have norm <= 1")
  session:run({}, {}, layer.updates)
  local u = session:run({}, { layer.u.read })[1]:totable()
  assert_true(u[1][1] == u[1][1], "u must remain finite")
  session:close()
end)

test("multiclass_f1 reports macro F1", function()
  local graph = tf.Graph.new()
  local predictions = tf.placeholder(graph, "float32", { 3, 3 }, "mcp")
  local labels = tf.placeholder(graph, "float32", { 3, 3 }, "mcl")
  local metric = tfa.metrics.multiclass_f1(graph, predictions, labels, 3)
  local session = tf.Session.new(graph)
  local value = session:run({
    [predictions] = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } },
    [labels] = { { 1, 0, 0 }, { 0, 1, 0 }, { 1, 0, 0 } },
  }, { metric })[1]:totable()
  assert_true(value > 0.6 and value < 0.9)
  session:close()
end)

test("multiclass f1 is one for perfect predictions", function()
  local graph = tf.Graph.new()
  local predictions = tf.placeholder(graph, "int32", { 4 }, "mf1_pred")
  local labels = tf.placeholder(graph, "float32", { 4, 3 }, "mf1_labels")
  local outputs = {
    tfa.metrics.f1_score(graph, predictions, labels,
      { average = "macro", num_classes = 3, name = "mf1_macro" }),
    tfa.metrics.f1_score(graph, predictions, labels,
      { average = "micro", num_classes = 3, name = "mf1_micro" }),
    tfa.metrics.f1_score(graph, predictions, labels,
      { average = "weighted", num_classes = 3, name = "mf1_weighted" }),
  }
  local session = tf.Session.new(graph)
  local values = session:run({
    [predictions] = { 0, 1, 2, 0 },
    [labels] = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 0 } },
  }, outputs)
  for index = 1, 3 do
    near(values[index]:totable(), 1, 1e-6, "perfect f1 " .. index)
  end
  session:close()
end)

test("multiclass f1 distinguishes macro, micro, and weighted averages",
    function()
  local graph = tf.Graph.new()
  local predictions = tf.placeholder(graph, "int32", { 5 }, "mf1_avg_pred")
  local labels = tf.placeholder(graph, "float32", { 5, 3 }, "mf1_avg_labels")
  local outputs = {
    tfa.metrics.f1_score(graph, predictions, labels,
      { average = "macro", name = "mf1_avg_macro" }),
    tfa.metrics.f1_score(graph, predictions, labels,
      { average = "micro", name = "mf1_avg_micro" }),
    tfa.metrics.f1_score(graph, predictions, labels,
      { average = "weighted", name = "mf1_avg_weighted" }),
  }
  local session = tf.Session.new(graph)
  local values = session:run({
    [predictions] = { 0, 0, 1, 2, 2 },
    [labels] = { { 1, 0, 0 }, { 1, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 },
      { 0, 0, 1 } },
  }, outputs)
  near(values[1]:totable(), 22 / 45, 1e-6, "macro f1")
  near(values[2]:totable(), 3 / 5, 1e-6, "micro f1")
  near(values[3]:totable(), 46 / 75, 1e-6, "weighted f1")
  session:close()
end)

test("multiclass f1 accepts one-hot predictions and class labels", function()
  local graph = tf.Graph.new()
  local predictions = tf.placeholder(graph, "float32", { 5, 3 },
    "mf1_hot_pred")
  local labels = tf.placeholder(graph, "int32", { 5 }, "mf1_class_labels")
  local metric = tfa.metrics.f1_score(graph, predictions, labels,
    { average = "weighted", num_classes = 3 })
  local values = run(graph, {
    [predictions] = { { 1, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 },
      { 0, 0, 1 }, { 0, 0, 1 } },
    [labels] = { 0, 0, 0, 1, 2 },
  }, metric)
  near(values, 46 / 75, 1e-6, "weighted one-hot f1")
end)

test("Cohen's kappa supports weighted multiclass confusion matrices",
    function()
  local graph = tf.Graph.new()
  local matrix = tf.placeholder(graph, "float32", { 3, 3 }, "ck_matrix")
  local predictions = tf.placeholder(graph, "int32", { 5 }, "ck_pred")
  local labels = tf.placeholder(graph, "int32", { 5 }, "ck_labels")
  local outputs = {
    tfa.metrics.cohen_kappa(graph, matrix,
      { weights = "none", name = "ck_none" }),
    tfa.metrics.cohen_kappa(graph, matrix,
      { weights = "linear", name = "ck_linear" }),
    tfa.metrics.cohen_kappa(graph, matrix,
      { weights = "quadratic", name = "ck_quadratic" }),
    tfa.metrics.cohen_kappa(graph, predictions, labels,
      { weights = "quadratic", name = "ck_from_classes" }),
  }
  local session = tf.Session.new(graph)
  local values = session:run({
    [matrix] = { { 2, 1, 0 }, { 0, 0, 1 }, { 0, 0, 1 } },
    [predictions] = { 0, 0, 1, 2, 2 },
    [labels] = { 0, 0, 0, 1, 2 },
  }, outputs)
  near(values[1]:totable(), 3 / 8, 1e-6, "unweighted kappa")
  near(values[2]:totable(), 7 / 12, 1e-6, "linear kappa")
  near(values[3]:totable(), 3 / 4, 1e-6, "quadratic kappa")
  near(values[4]:totable(), 3 / 4, 1e-6, "kappa from class labels")
  session:close()
end)

test("multiclass metrics reject invalid options and shapes", function()
  local graph = tf.Graph.new()
  local predictions = tf.placeholder(graph, "float32", { 2 }, "inv_pred")
  local labels = tf.placeholder(graph, "float32", { 2 }, "inv_labels")
  local invalid_average = pcall(tfa.metrics.f1_score, graph, predictions,
    labels, { average = "median" })
  assert_true(not invalid_average, "invalid f1 average should fail")
  local invalid_classes = pcall(tfa.metrics.f1_score, graph, predictions,
    labels, { average = "macro", num_classes = 1 })
  assert_true(not invalid_classes, "invalid num_classes should fail")
  local invalid_weights = pcall(tfa.metrics.cohen_kappa, graph, predictions,
    labels, { weights = "cubic" })
  assert_true(not invalid_weights, "invalid kappa weights should fail")

  local mismatched = pcall(function()
    local bad_graph = tf.Graph.new()
    local bad_predictions = tf.placeholder(bad_graph, "int32", { 3 },
      "bad_pred")
    local bad_labels = tf.placeholder(bad_graph, "float32", { 2, 3 },
      "bad_labels")
    local bad_metric = tfa.metrics.f1_score(bad_graph, bad_predictions,
      bad_labels, { average = "macro", num_classes = 3 })
    local session = tf.Session.new(bad_graph)
    session:run({ [bad_predictions] = { 0, 1, 2 },
      [bad_labels] = { { 1, 0, 0 }, { 0, 1, 0 } } }, { bad_metric })
    session:close()
  end)
  assert_true(not mismatched, "mismatched prediction and label shapes fail")

  local nonsquare = pcall(function()
    local matrix_graph = tf.Graph.new()
    local matrix = tf.placeholder(matrix_graph, "float32", { 2, 3 },
      "bad_matrix")
    local bad_kappa = tfa.metrics.cohen_kappa(matrix_graph, matrix,
      { weights = "quadratic" })
    local session = tf.Session.new(matrix_graph)
    session:run({ [matrix] = { { 1, 0, 0 }, { 0, 1, 0 } } }, { bad_kappa })
    session:close()
  end)
  assert_true(not nonsquare, "non-square confusion matrix should fail")
end)

test("image flips and brightness adjust preserve shape", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 1, 2, 3, 1 }, "img")
  local left = tfa.image.flip_left_right(graph, input, "flip_lr")
  local up = tfa.image.flip_up_down(graph, input, "flip_ud")
  local bright = tfa.image.adjust_brightness(graph, input, 0.5, "bright")
  local session = tf.Session.new(graph)
  local output = session:run({ [input] = { { { { 1 }, { 2 }, { 3 } }, { { 4 }, { 5 }, { 6 } } } } },
    { left, up, bright })[1]
  assert_eq(table.concat(output:shape(), ","), "1,2,3,1")
  session:close()
end)

test("weight decay wrapper and quantization layer build graphs", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 2, 2 }, "qin")
  local quant = tfa.layers.quantization(graph, input, { name = "q" })
  local weight = tf.variable(graph, { { 1, 1 }, { 1, 1 } }, "w")
  local loss = tf.reduce_sum(graph, tf.mul(graph, quant, weight.read,
    "weighted", "float32"), { 0, 1 }, "qloss", "float32")
  local optimizer = tfa.optimizers.WeightDecay("adam", { weight_decay = 0.01 })
  local updates = optimizer(graph, loss, { weight })
  assert_true(#updates > 0)
  local session = tf.Session.new(graph)
  session:run({}, {}, { weight.initializer })
  local output = session:run({ [input] = { { 1, 2 }, { 3, 4 } } }, { quant })[1]
  assert_eq(table.concat(output:shape(), ","), "2,2")
  session:close()
end)

test("LazyAdam only updates elements with non-zero gradients", function()
  local graph = tf.Graph.new()
  local weight = tf.variable(graph, { { 1, 2 }, { 3, 4 } }, "lazy_weight",
    { dtype = "float32" })
  local mask = tf.constant(graph, { { 1, 0 }, { 0, 1 } }, "lazy_mask",
    "float32")
  local loss = tf.reduce_sum(graph, tf.mul(graph, weight.read, mask,
    "lazy_weighted", "float32"), { 0, 1 }, "lazy_loss", "float32")
  local targets, gradients, state = tfa.optimizers.LazyAdam(graph, loss,
    { weight }, { learning_rate = 0.1 })
  assert_eq(#gradients, 1)
  local session = tf.Session.new(graph)
  session:run({}, {}, { weight.initializer, unpack(state.initializers) })
  session:run({}, {}, targets)
  session:run({}, {}, targets)
  local values = session:run({}, { weight.read })[1]:totable()
  near(values[1][1], 0.8, 1e-6, "lazy weight (1,1)")
  near(values[1][2], 2, 1e-6, "lazy untouched (1,2)")
  near(values[2][1], 3, 1e-6, "lazy untouched (2,1)")
  near(values[2][2], 3.8, 1e-6, "lazy weight (2,2)")
  local moments = session:run({}, { state.variables[3].read })[1]:totable()
  near(moments[1][1], 0.19, 1e-6, "lazy first moment")
  near(moments[1][2], 0, 1e-6, "lazy untouched moment")
  local beta1_power = session:run({}, { state.beta1_power.read })[1]:totable()
  near(beta1_power, 0.729, 1e-6, "lazy beta1 power")
  session:close()
end)

test("LazyAdam rejects an empty variable list", function()
  local ok, err = pcall(tfa.optimizers.LazyAdam, tf.Graph.new(), nil, {})
  assert_true(not ok, "LazyAdam should reject an empty variable list")
  assert_true(tostring(err):find("non-empty", 1, true) ~= nil,
    "LazyAdam error should mention a non-empty variable list")
end)

test("log_cosh matches the stable formula and reductions", function()
  local graph = tf.Graph.new()
  local labels = tf.placeholder(graph, "float32", { 2 }, "lc_labels")
  local predictions = tf.placeholder(graph, "float32", { 2 }, "lc_pred")
  local elementwise = tfa.losses.log_cosh(graph, labels, predictions,
    { reduction = "none", name = "lc_none" })
  local mean = tfa.losses.log_cosh(graph, labels, predictions,
    { reduction = "mean", name = "lc_mean" })
  local total = tfa.losses.log_cosh(graph, labels, predictions,
    { reduction = "sum", name = "lc_sum" })
  local session = tf.Session.new(graph)
  local values = session:run({
    [labels] = { 0, 0 },
    [predictions] = { 1, 1000 },
  }, { elementwise, mean, total })
  local single = math.log(math.cosh(1))
  local large = 1000 - math.log(2)
  local elements = values[1]:totable()
  near(elements[1], single, 1e-6, "log_cosh small residual")
  near(elements[2], large, 1e-3, "log_cosh large residual")
  near(values[2]:totable(), (single + large) / 2, 1e-3, "log_cosh mean")
  near(values[3]:totable(), single + large, 1e-3, "log_cosh sum")
  session:close()
end)

test("log_cosh rejects an unknown reduction", function()
  local graph = tf.Graph.new()
  local labels = tf.placeholder(graph, "float32", { 1 }, "lc_bad_labels")
  local predictions = tf.placeholder(graph, "float32", { 1 }, "lc_bad_pred")
  local ok = pcall(tfa.losses.log_cosh, graph, labels, predictions,
    { reduction = "median" })
  assert_true(not ok, "log_cosh should reject an unknown reduction")
end)

test("mean_iou averages IoU over classes present in the batch", function()
  local graph = tf.Graph.new()
  local predictions = tf.placeholder(graph, "int32", { 4 }, "miou_pred")
  local labels = tf.placeholder(graph, "int32", { 4 }, "miou_labels")
  local metric = tfa.metrics.mean_iou(graph, predictions, labels, 4,
    { name = "miou" })
  local session = tf.Session.new(graph)
  local value = session:run({
    [predictions] = { 0, 1, 2, 2 },
    [labels] = { 0, 1, 2, 0 },
  }, { metric })[1]:totable()
  near(value, 2 / 3, 1e-6, "mean_iou")
  session:close()
end)

test("mean_iou accepts one-hot rows and exposes the MeanIoU alias", function()
  local graph = tf.Graph.new()
  local predictions = tf.placeholder(graph, "float32", { 3, 3 }, "miou_hot")
  local labels = tf.placeholder(graph, "float32", { 3, 3 }, "miou_hot_labels")
  local metric = tfa.metrics.MeanIoU(graph, predictions, labels,
    { num_classes = 3, name = "miou_hot_metric" })
  local value = run(graph, {
    [predictions] = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } },
    [labels] = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } },
  }, metric)
  near(value, 1, 1e-6, "one-hot mean_iou")
end)

test("gaussian_filter2d preserves mass and smooths an impulse", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 1, 3, 3, 1 }, "gauss_in")
  local filtered = tfa.image.gaussian_filter2d(graph, input,
    { filter_shape = { 3, 3 }, sigma = 1, padding = "CONSTANT",
      constant_values = 0, name = "gauss" })
  local session = tf.Session.new(graph)
  local tensor = session:run({ [input] = { {
    { { 0 }, { 0 }, { 0 } },
    { { 0 }, { 1 }, { 0 } },
    { { 0 }, { 0 }, { 0 } },
  } } }, { filtered })[1]
  assert_eq(table.concat(tensor:shape(), ","), "1,3,3,1")
  local values = tensor:totable()
  local expected_center = 1 / (1 + 4 * math.exp(-0.5) + 4 * math.exp(-1))
  near(values[1][2][2][1], expected_center, 1e-6, "gaussian center")
  local total = 0
  for _, row in ipairs(values[1]) do
    for _, pixel in ipairs(row) do total = total + pixel[1] end
  end
  near(total, 1, 1e-5, "gaussian mass")
  session:close()
end)

test("gaussian_filter2d supports rank-3 images with reflect padding",
    function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 3, 3, 2 }, "gauss_hwc")
  local filtered = tfa.image.gaussian_filter2d(graph, input,
    { rank = 3, filter_shape = { 3, 3 }, sigma = 1, name = "gauss_hwc_out" })
  local session = tf.Session.new(graph)
  local tensor = session:run({ [input] = {
    { { 1, 1 }, { 1, 1 }, { 1, 1 } },
    { { 1, 1 }, { 1, 1 }, { 1, 1 } },
    { { 1, 1 }, { 1, 1 }, { 1, 1 } },
  } }, { filtered })[1]
  assert_eq(table.concat(tensor:shape(), ","), "3,3,2")
  local values = tensor:totable()
  for row = 1, 3 do
    for column = 1, 3 do
      for channel = 1, 2 do
        near(values[row][column][channel], 1, 1e-6, "gaussian constant image")
      end
    end
  end
  session:close()
end)

test("maxout selects the maximum of each channel group", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 1, 2, 4 }, "maxout_in")
  local output = tfa.layers.maxout(graph, input, 2, { name = "maxout" })
  local session = tf.Session.new(graph)
  local tensor = session:run({ [input] = { {
    { 1, 2, 3, 4 },
    { 5, 6, 7, 8 },
  } } }, { output })[1]
  assert_eq(table.concat(tensor:shape(), ","), "1,2,2")
  local values = tensor:totable()
  near(values[1][1][1], 2, 0, "maxout (1,1)")
  near(values[1][1][2], 4, 0, "maxout (1,2)")
  near(values[1][2][1], 6, 0, "maxout (2,1)")
  near(values[1][2][2], 8, 0, "maxout (2,2)")
  session:close()
end)

test("maxout supports a non-last axis and rejects indivisible groups",
    function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 2, 4, 2 }, "maxout_axis_in")
  local output = tfa.layers.Maxout(graph, input, 2,
    { axis = 1, input_shape = { 2, 4, 2 }, name = "maxout_axis" })
  local session = tf.Session.new(graph)
  local values = session:run({ [input] = {
    { { 1, 10 }, { 2, 20 }, { 3, 30 }, { 4, 40 } },
    { { 5, 50 }, { 6, 60 }, { 7, 70 }, { 8, 80 } },
  } }, { output })[1]:totable()
  near(values[1][1][1], 2, 0, "maxout axis batch0 group0 channel0")
  near(values[1][1][2], 20, 0, "maxout axis batch0 group0 channel1")
  near(values[1][2][1], 4, 0, "maxout axis batch0 group1 channel0")
  near(values[1][2][2], 40, 0, "maxout axis batch0 group1 channel1")
  near(values[2][1][1], 6, 0, "maxout axis batch1 group0 channel0")
  near(values[2][2][2], 80, 0, "maxout axis batch1 group1 channel1")
  local ok = pcall(tfa.layers.maxout, graph, input, 3,
    { axis = 0, input_shape = { 2, 4, 2 } })
  assert_true(not ok, "maxout should reject indivisible axis sizes")
  session:close()
end)

test("poisson returns the rate and applies inverse-CDF noise", function()
  local graph = tf.Graph.new()
  local log_rate = tf.placeholder(graph, "float32", { 2 }, "poisson_rate_in")
  local noise = tf.placeholder(graph, "float32", { 2 }, "poisson_noise_in")
  local deterministic = tfa.layers.poisson(graph, log_rate,
    { name = "poisson_rate" })
  local sampled = tfa.layers.poisson(graph, log_rate,
    { noise = noise, name = "poisson_sample" })
  local session = tf.Session.new(graph)
  local values = session:run({
    [log_rate] = { 0, math.log(2) },
    [noise] = { 0.5, 0.5 },
  }, { deterministic, sampled })
  near(values[1]:totable()[1], 1, 1e-6, "poisson rate zero")
  near(values[1]:totable()[2], 2, 1e-6, "poisson rate log two")
  near(values[2]:totable()[1], -math.log(0.5), 1e-6, "poisson noise")
  near(values[2]:totable()[2], -math.log(0.5), 1e-6, "poisson noise")
  session:close()
end)

test("pinball loss matches the quantile formula and reductions", function()
  local graph = tf.Graph.new()
  local labels = tf.placeholder(graph, "float32", { 4 }, "pinball_labels")
  local predictions = tf.placeholder(graph, "float32", { 4 },
    "pinball_predictions")
  local outputs = {
    tfa.losses.pinball(graph, labels, predictions,
      { tau = 0.1, reduction = "none", name = "pinball_none" }),
    tfa.losses.pinball(graph, labels, predictions,
      { tau = 0.1, reduction = "mean", name = "pinball_mean" }),
    tfa.losses.pinball(graph, labels, predictions,
      { tau = 0.1, reduction = "sum", name = "pinball_sum" }),
  }
  local session = tf.Session.new(graph)
  local values = session:run({
    [labels] = { 0, 0, 1, 1 },
    [predictions] = { 1, 1, 1, 0 },
  }, outputs)
  local elementwise = values[1]:totable()
  local expected = { 0.9, 0.9, 0, 0.1 }
  for index = 1, 4 do
    near(elementwise[index], expected[index], 1e-6,
      "pinball element " .. index)
  end
  near(values[2]:totable(), 0.475, 1e-6, "pinball mean")
  near(values[3]:totable(), 1.9, 1e-6, "pinball sum")
  assert_true(tfa.losses.PinballLoss == tfa.losses.pinball)
  assert_true(tfa.losses.pinball_loss == tfa.losses.pinball)
  session:close()
end)

test("sigmoid focal cross entropy matches Addons values and logits", function()
  local graph = tf.Graph.new()
  local labels = tf.placeholder(graph, "float32", { 3, 1 },
    "sigmoid_focal_labels")
  local predictions = tf.placeholder(graph, "float32", { 3, 1 },
    "sigmoid_focal_predictions")
  local elementwise = tfa.losses.sigmoid_focal_crossentropy(graph, labels,
    predictions, { reduction = "none", name = "sigmoid_focal_none" })
  local mean = tfa.losses.sigmoid_focal_crossentropy(graph, labels,
    predictions, { reduction = "mean", name = "sigmoid_focal_mean" })
  local total = tfa.losses.sigmoid_focal_crossentropy(graph, labels,
    predictions, { reduction = "sum", name = "sigmoid_focal_sum" })
  local logits = tf.constant(graph, { { 3.4760986898352724 },
    { 2.313634929180631 }, { -3.4760986898352733 } }, "sigmoid_focal_logits",
    "float32")
  local from_logits = tfa.losses.sigmoid_focal_crossentropy(graph, labels,
    logits, { from_logits = true, reduction = "none",
      name = "sigmoid_focal_logits" })
  local session = tf.Session.new(graph)
  local values = session:run({
    [labels] = { { 1 }, { 1 }, { 0 } },
    [predictions] = { { 0.97 }, { 0.91 }, { 0.03 } },
  }, { elementwise, mean, total, from_logits })
  local expected = { 6.853321684059441e-06, 1.909791259292635e-04,
    2.0559965052178325e-05 }
  local actual = values[1]:totable()
  for index = 1, 3 do
    near(actual[index][1], expected[index], 1e-8,
      "sigmoid focal element " .. index)
    near(values[4]:totable()[index][1], expected[index], 1e-6,
      "sigmoid focal logits element " .. index)
  end
  near(values[2]:totable(),
    (expected[1] + expected[2] + expected[3]) / 3, 1e-8, "sigmoid focal mean")
  near(values[3]:totable(), expected[1] + expected[2] + expected[3], 1e-8,
    "sigmoid focal sum")
  session:close()
end)

test("multilabel confusion matrix matches the Addons layout", function()
  local graph = tf.Graph.new()
  local predictions = tf.placeholder(graph, "float32", { 2, 3 },
    "mlcm_predictions")
  local labels = tf.placeholder(graph, "float32", { 2, 3 }, "mlcm_labels")
  local logits = tf.constant(graph, { { 100, -100, -100 },
    { -100, 100, 100 } }, "mlcm_logits", "float32")
  local metric = tfa.metrics.multilabel_confusion_matrix(graph, predictions,
    labels, 3, { name = "mlcm" })
  local alias = tfa.metrics.MultiLabelConfusionMatrix(graph, predictions,
    labels, { num_classes = 3, name = "mlcm_alias" })
  local thresholded = tfa.metrics.multilabel_confusion_matrix(graph,
    predictions, labels, 3, { threshold = 0.5, name = "mlcm_thresholded" })
  local from_logits = tfa.metrics.multilabel_confusion_matrix(graph, logits,
    labels, 3, { from_logits = true, threshold = 0.5,
      name = "mlcm_from_logits" })
  local session = tf.Session.new(graph)
  local values = session:run({
    [labels] = { { 1, 0, 1 }, { 0, 1, 0 } },
    [predictions] = { { 1, 0, 0 }, { 0, 1, 1 } },
  }, { metric, alias, thresholded, from_logits })
  local expected = {
    { { 1, 0 }, { 0, 1 } },
    { { 1, 0 }, { 0, 1 } },
    { { 0, 1 }, { 1, 0 } },
  }
  for class = 1, 3 do
    for row = 1, 2 do
      for column = 1, 2 do
        near(values[1]:totable()[class][row][column],
          expected[class][row][column], 0,
          string.format("mlcm class %d cell %d,%d", class, row, column))
        near(values[2]:totable()[class][row][column],
          expected[class][row][column], 0,
          string.format("mlcm alias class %d cell %d,%d", class, row, column))
        near(values[3]:totable()[class][row][column],
          expected[class][row][column], 0,
          string.format("mlcm threshold class %d cell %d,%d", class, row,
            column))
        near(values[4]:totable()[class][row][column],
          expected[class][row][column], 0,
          string.format("mlcm logits class %d cell %d,%d", class, row, column))
      end
    end
  end
  assert_true(tfa.metrics.MultiLabelConfusionMatrix ==
    tfa.metrics.multilabel_confusion_matrix)
  session:close()
end)

test("median_filter2d removes impulses and supports even filters", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 1, 3, 3, 1 }, "median_in")
  local impulse = tfa.image.median_filter2d(graph, input,
    { filter_shape = { 3, 3 }, padding = "REFLECT", name = "median_impulse" })
  local ramp = tfa.image.median_filter2d(graph, input,
    { filter_shape = { 2, 2 }, padding = "CONSTANT", constant_values = 0,
      name = "median_ramp" })
  local session = tf.Session.new(graph)
  local impulse_tensor = session:run({ [input] = { {
    { { 0 }, { 0 }, { 0 } },
    { { 0 }, { 1 }, { 0 } },
    { { 0 }, { 0 }, { 0 } },
  } } }, { impulse })[1]
  assert_eq(table.concat(impulse_tensor:shape(), ","), "1,3,3,1")
  local impulse_values = impulse_tensor:totable()
  for row = 1, 3 do
    for column = 1, 3 do
      near(impulse_values[1][row][column][1], 0, 0,
        "median impulse removal")
    end
  end
  local ramp_tensor = session:run({ [input] = { {
    { { 1 }, { 2 }, { 3 } },
    { { 4 }, { 5 }, { 6 } },
    { { 7 }, { 8 }, { 9 } },
  } } }, { ramp })[1]
  local ramp_values = ramp_tensor:totable()
  near(ramp_values[1][1][1][1], 3, 0, "median even filter top-left")
  near(ramp_values[1][2][2][1], 7, 0, "median even filter center")
  session:close()
end)

test("median_filter2d preserves rank-2 and rank-3 constant images", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 3, 3, 2 }, "median_hwc")
  local grayscale = tf.placeholder(graph, "float32", { 3, 3 }, "median_hw")
  local filtered = tfa.image.median_filter2d(graph, input,
    { rank = 3, filter_shape = { 3, 3 }, name = "median_hwc_out" })
  local flat_filtered = tfa.image.median_filter2d(graph, grayscale,
    { rank = 2, filter_shape = { 3, 3 }, name = "median_hw_out" })
  local session = tf.Session.new(graph)
  local tensor, flat_tensor = unpack(session:run({
    [input] = {
      { { 1, 1 }, { 1, 1 }, { 1, 1 } },
      { { 1, 1 }, { 1, 1 }, { 1, 1 } },
      { { 1, 1 }, { 1, 1 }, { 1, 1 } },
    },
    [grayscale] = { { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 } },
  }, { filtered, flat_filtered }))
  assert_eq(table.concat(tensor:shape(), ","), "3,3,2")
  local values = tensor:totable()
  for row = 1, 3 do
    for column = 1, 3 do
      for channel = 1, 2 do
        near(values[row][column][channel], 1, 0,
          "median rank-3 constant image")
      end
    end
  end
  assert_eq(table.concat(flat_tensor:shape(), ","), "3,3")
  local flat_values = flat_tensor:totable()
  for row = 1, 3 do
    for column = 1, 3 do
      near(flat_values[row][column], 1, 0, "median rank-2 constant image")
    end
  end
  session:close()
end)

test("stochastic depth honors survival probability and training", function()
  local graph = tf.Graph.new()
  local shortcut = tf.placeholder(graph, "float32", { 2, 2 }, "sd_shortcut")
  local residual = tf.placeholder(graph, "float32", { 2, 2 }, "sd_residual")
  local training = tf.placeholder(graph, "bool", {}, "sd_training")
  local inference = tfa.layers.stochastic_depth(graph, shortcut, residual,
    { survival_probability = 0.5, training = false, name = "sd_inference" })
  local dropped = tfa.layers.stochastic_depth(graph, shortcut, residual,
    { survival_probability = 0, training = true, name = "sd_dropped" })
  local kept = tfa.layers.stochastic_depth(graph, shortcut, residual,
    { survival_probability = 1, training = true, name = "sd_kept" })
  local dynamic = tfa.layers.stochastic_depth(graph, shortcut, residual,
    { survival_probability = 0.5, training = training, seed = 7, seed2 = 11,
      name = "sd_dynamic" })
  local session = tf.Session.new(graph)
  local feeds = {
    [shortcut] = { { 1, 1 }, { 1, 1 } },
    [residual] = { { 1, 1 }, { 1, 1 } },
  }
  feeds[training] = false
  local values = session:run(feeds, { inference, dropped, kept, dynamic })
  near(values[1]:totable()[1][1], 1.5, 1e-6, "stochastic depth inference")
  near(values[2]:totable()[1][1], 1, 0, "stochastic depth dropped")
  near(values[3]:totable()[1][1], 2, 0, "stochastic depth kept")
  near(values[4]:totable()[1][1], 1.5, 1e-6,
    "stochastic depth dynamic inference")
  feeds[training] = true
  local train_value = session:run(feeds, { dynamic })[1]:totable()[1][1]
  assert_true(math.abs(train_value - 1.5) > 1e-6,
    "stochastic depth training should use a Bernoulli residual mask")
  session:close()
end)

test("npairs loss matches the normalized-adjacency softmax formula", function()
  local graph = tf.Graph.new()
  local labels = tf.placeholder(graph, "int32", { 4 }, "np_labels")
  local similarity = tf.placeholder(graph, "float32", { 4, 4 },
    "np_similarity")
  local multilabel = tf.placeholder(graph, "float32", { 4, 2 },
    "np_multilabel")
  local loss = tfa.losses.npairs_loss(graph, labels, similarity)
  local multilabel_loss = tfa.losses.npairs_multilabel_loss(graph, multilabel,
    similarity, { name = "npairs_multilabel" })
  local session = tf.Session.new(graph)
  local matrix = {
    { 2, 1, 0, 0 },
    { 1, 2, 0, 0 },
    { 0, 0, 2, 1 },
    { 0, 0, 1, 2 },
  }
  local function row_loss(row, first, second)
    local maximum = row[1]
    for index = 2, 4 do maximum = math.max(maximum, row[index]) end
    local total = 0
    for index = 1, 4 do total = total + math.exp(row[index] - maximum) end
    local log_total = math.log(total)
    local value = 0
    for _, index in ipairs({ first, second }) do
      value = value - 0.5 * (row[index] - maximum - log_total)
    end
    return value
  end
  local expected = (row_loss(matrix[1], 1, 2) + row_loss(matrix[2], 1, 2) +
    row_loss(matrix[3], 3, 4) + row_loss(matrix[4], 3, 4)) / 4
  local values = session:run({
    [labels] = { 0, 0, 1, 1 },
    [multilabel] = { { 1, 0 }, { 1, 0 }, { 0, 1 }, { 0, 1 } },
    [similarity] = matrix,
  }, { loss, multilabel_loss })
  near(values[1]:totable(), expected, 1e-6, "npairs loss")
  near(values[2]:totable(), expected, 1e-6, "npairs multilabel loss")
  assert_true(tfa.losses.NpairsLoss == tfa.losses.npairs_loss)
  assert_true(tfa.losses.NpairsMultilabelLoss ==
    tfa.losses.npairs_multilabel_loss)
  session:close()
end)

test("triplet semihard loss matches TensorFlow Addons examples", function()
  local graph = tf.Graph.new()
  local labels = tf.placeholder(graph, "int32", { 2 }, "tsh_labels")
  local embeddings = tf.placeholder(graph, "float32", { 2, 2 },
    "tsh_embeddings")
  local l2 = tfa.losses.triplet_semihard_loss(graph, labels, embeddings,
    { margin = 1, distance_metric = "L2", name = "tsh_l2" })
  local squared = tfa.losses.triplet_semihard_loss(graph, labels, embeddings,
    { margin = 1, distance_metric = "squared-L2", name = "tsh_squared" })
  local angular = tfa.losses.triplet_semihard_loss(graph, labels, embeddings,
    { margin = 1, distance_metric = "angular", name = "tsh_angular" })
  local session = tf.Session.new(graph)
  local values = session:run({
    [labels] = { 0, 0 },
    [embeddings] = { { 0, 1 }, { 1, 0 } },
  }, { l2, squared, angular })
  near(values[1]:totable(), 1 + math.sqrt(2), 1e-6, "semihard L2")
  near(values[2]:totable(), 3, 1e-6, "semihard squared L2")
  near(values[3]:totable(), 2, 1e-6, "semihard angular")
  assert_true(tfa.losses.TripletSemiHardLoss ==
    tfa.losses.triplet_semihard_loss)
  session:close()
end)

test("triplet semihard mines the closest larger negative", function()
  local graph = tf.Graph.new()
  local labels = tf.placeholder(graph, "int32", { 6 }, "tsh_mine_labels")
  local embeddings = tf.placeholder(graph, "float32", { 6, 1 },
    "tsh_mine_embeddings")
  local l2 = tfa.losses.triplet_semihard_loss(graph, labels, embeddings,
    { margin = 1, distance_metric = "L2", name = "tsh_mine_l2" })
  local squared = tfa.losses.triplet_semihard_loss(graph, labels, embeddings,
    { margin = 1, distance_metric = "squared-L2", name = "tsh_mine_squared" })
  local session = tf.Session.new(graph)
  local values = session:run({
    [labels] = { 0, 0, 1, 1, 2, 2 },
    [embeddings] = { { 0 }, { 1 }, { 2 }, { 3 }, { 4 }, { 8 } },
  }, { l2, squared })
  near(values[1]:totable(), 1 / 6, 1e-6, "semihard mined L2")
  near(values[2]:totable(), 1 / 6, 1e-6, "semihard mined squared L2")
  session:close()
end)

test("filter response normalization normalizes spatial responses", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 1, 2, 2, 2 }, "frn_input")
  local layer = tfa.layers.filter_response_normalization(graph, input,
    { input_shape = { 1, 2, 2, 2 }, name = "frn" })
  local learned = tfa.layers.filter_response_normalization(graph, input,
    { input_shape = { 1, 2, 2, 2 }, learned_epsilon = true,
      name = "frn_learned" })
  local session = tf.Session.new(graph)
  session:run({}, {}, { unpack(layer.initializers) })
  session:run({}, {}, { unpack(learned.initializers) })
  local values = session:run({ [input] = { {
    { { 1, 2 }, { 2, 2 } },
    { { 3, 2 }, { 4, 2 } },
  } } }, { layer.output, learned.output })
  local output = values[1]:totable()
  assert_eq(table.concat(values[1]:shape(), ","), "1,2,2,2")
  local first_denominator = math.sqrt(7.5 + 1e-6)
  near(output[1][1][1][1], 1 / first_denominator, 1e-6, "frn value 1")
  near(output[1][1][2][1], 2 / first_denominator, 1e-6, "frn value 2")
  near(output[1][2][1][1], 3 / first_denominator, 1e-6, "frn value 3")
  near(output[1][2][2][1], 4 / first_denominator, 1e-6, "frn value 4")
  local second_denominator = math.sqrt(4 + 1e-6)
  near(output[1][1][1][2], 2 / second_denominator, 1e-6, "frn channel two")
  assert_true(values[2]:totable()[1][1][1][1] ==
    values[2]:totable()[1][1][1][1], "learned epsilon produced NaN")
  assert_true(tfa.layers.FilterResponseNormalization ==
    tfa.layers.filter_response_normalization)
  session:close()
end)

test("hamming loss matches the multiclass and multilabel examples", function()
  local graph = tf.Graph.new()
  local multiclass_labels = tf.placeholder(graph, "float32", { 4, 4 },
    "hamming_mc_labels")
  local multiclass_predictions = tf.placeholder(graph, "float32", { 4, 4 },
    "hamming_mc_predictions")
  local multilabel_labels = tf.placeholder(graph, "int32", { 3, 4 },
    "hamming_ml_labels")
  local multilabel_predictions = tf.placeholder(graph, "float32", { 3, 4 },
    "hamming_ml_predictions")
  local actuals = tf.placeholder(graph, "int32", { 4 }, "hamming_actuals")
  local guesses = tf.placeholder(graph, "int32", { 4 }, "hamming_guesses")
  local multiclass = tfa.metrics.hamming_loss(graph, multiclass_predictions,
    multiclass_labels, { mode = "multiclass", threshold = 0.6,
      name = "hamming_mc" })
  local multilabel = tfa.metrics.hamming_loss(graph, multilabel_predictions,
    multilabel_labels, { mode = "multilabel", threshold = 0.8,
      name = "hamming_ml" })
  local distance = tfa.metrics.hamming_distance(graph, actuals, guesses,
    { name = "hamming_distance" })
  local session = tf.Session.new(graph)
  local values = session:run({
    [multiclass_labels] = {
      { 1, 0, 0, 0 }, { 0, 0, 1, 0 }, { 0, 0, 0, 1 }, { 0, 1, 0, 0 },
    },
    [multiclass_predictions] = {
      { 0.8, 0.1, 0.1, 0.0 }, { 0.2, 0.0, 0.8, 0.0 },
      { 0.05, 0.05, 0.1, 0.8 }, { 1.0, 0.0, 0.0, 0.0 },
    },
    [multilabel_labels] = { { 1, 0, 1, 0 }, { 0, 1, 0, 1 },
      { 0, 0, 0, 1 } },
    [multilabel_predictions] = { { 0.82, 0.5, 0.90, 0 }, { 0, 1, 0.4, 0.98 },
      { 0.89, 0.79, 0, 0.3 } },
    [actuals] = { 1, 1, 0, 0 },
    [guesses] = { 1, 0, 0, 0 },
  }, { multiclass, multilabel, distance })
  near(values[1]:totable(), 0.25, 1e-6, "multiclass hamming loss")
  near(values[2]:totable(), 1 / 6, 1e-6, "multilabel hamming loss")
  near(values[3]:totable(), 0.25, 1e-6, "hamming distance")
  assert_true(tfa.metrics.HammingLoss == tfa.metrics.hamming_loss)
  session:close()
end)

test("mean_filter2d preserves mass and supports even filters", function()
  local graph = tf.Graph.new()
  local impulse = tf.placeholder(graph, "float32", { 1, 3, 3, 1 },
    "mean_impulse")
  local ramp = tf.placeholder(graph, "float32", { 3, 3 }, "mean_ramp")
  local flat = tf.placeholder(graph, "float32", { 3, 3, 2 }, "mean_flat")
  local blurred = tfa.image.mean_filter2d(graph, impulse,
    { filter_shape = { 3, 3 }, padding = "CONSTANT", constant_values = 0,
      name = "mean_impulse_out" })
  local even = tfa.image.mean_filter2d(graph, ramp,
    { rank = 2, filter_shape = { 2, 2 }, padding = "CONSTANT",
      constant_values = 0, name = "mean_even_out" })
  local channel = tfa.image.mean_filter2d(graph, flat,
    { rank = 3, filter_shape = { 3, 3 }, padding = "REFLECT",
      name = "mean_channel_out" })
  local session = tf.Session.new(graph)
  local values = session:run({
    [impulse] = { {
      { { 0 }, { 0 }, { 0 } },
      { { 0 }, { 1 }, { 0 } },
      { { 0 }, { 0 }, { 0 } },
    } },
    [ramp] = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } },
    [flat] = {
      { { 1, 1 }, { 1, 1 }, { 1, 1 } },
      { { 1, 1 }, { 1, 1 }, { 1, 1 } },
      { { 1, 1 }, { 1, 1 }, { 1, 1 } },
    },
  }, { blurred, even, channel })
  assert_eq(table.concat(values[1]:shape(), ","), "1,3,3,1")
  local impulse_values = values[1]:totable()
  local total = 0
  for row = 1, 3 do
    for column = 1, 3 do
      near(impulse_values[1][row][column][1], 1 / 9, 1e-6,
        "mean impulse value")
      total = total + impulse_values[1][row][column][1]
    end
  end
  near(total, 1, 1e-6, "mean impulse mass")
  local even_values = values[2]:totable()
  near(even_values[1][1], 3, 1e-6, "mean even top-left")
  near(even_values[2][2], 7, 1e-6, "mean even center")
  assert_eq(table.concat(values[3]:shape(), ","), "3,3,2")
  for row = 1, 3 do
    for column = 1, 3 do
      for channel_index = 1, 2 do
        near(values[3]:totable()[row][column][channel_index], 1, 1e-6,
          "mean rank-3 constant image")
      end
    end
  end
  session:close()
end)

test("hinge loss matches the margin formula and reductions", function()
  local graph = tf.Graph.new()
  local labels = tf.placeholder(graph, "float32", { 4 }, "hinge_labels")
  local predictions = tf.placeholder(graph, "float32", { 4 },
    "hinge_predictions")
  local elementwise = tfa.losses.hinge(graph, labels, predictions,
    { reduction = "none", name = "hinge_none" })
  local mean = tfa.losses.hinge(graph, labels, predictions,
    { name = "hinge_mean" })
  local total = tfa.losses.hinge(graph, labels, predictions,
    { reduction = "sum", name = "hinge_sum" })
  local custom = tfa.losses.hinge(graph, labels, predictions,
    { margin = 0.5, name = "hinge_custom" })
  local session = tf.Session.new(graph)
  local values = session:run({
    [labels] = { -1, 1, 1, -1 },
    [predictions] = { 0.5, 2, -0.5, 1 },
  }, { elementwise, mean, total, custom })
  local expected = { 1.5, 0, 1.5, 2 }
  local actual = values[1]:totable()
  for index = 1, 4 do
    near(actual[index], expected[index], 1e-6,
      "hinge element " .. index)
  end
  near(values[2]:totable(), 1.25, 1e-6, "hinge mean")
  near(values[3]:totable(), 5, 1e-6, "hinge sum")
  near(values[4]:totable(), 0.875, 1e-6, "hinge custom margin")
  assert_true(tfa.losses.Hinge == tfa.losses.hinge)
  assert_true(tfa.losses.hinge_loss == tfa.losses.hinge)
  session:close()
end)

test("KL divergence matches the last-axis formula and reductions", function()
  local graph = tf.Graph.new()
  local labels = tf.placeholder(graph, "float32", { 3, 3 }, "kl_labels")
  local predictions = tf.placeholder(graph, "float32", { 3, 3 },
    "kl_predictions")
  local elementwise = tfa.losses.kl_divergence(graph, labels, predictions,
    { reduction = "none", name = "kl_none" })
  local mean = tfa.losses.kl_divergence(graph, labels, predictions,
    { name = "kl_mean" })
  local total = tfa.losses.kl_divergence(graph, labels, predictions,
    { reduction = "sum", name = "kl_sum" })
  local session = tf.Session.new(graph)
  local values = session:run({
    [labels] = { { 0.5, 0.5, 0 }, { 0, 1, 0 }, { 0, 0, 0 } },
    [predictions] = { { 0.25, 0.75, 0 }, { 0, 0.5, 0.5 },
      { 0, 0, 0 } },
  }, { elementwise, mean, total })
  local first = 0.5 * math.log(2) + 0.5 * math.log(2 / 3)
  local second = math.log(2)
  local actual = values[1]:totable()
  near(actual[1], first, 1e-6, "KL first example")
  near(actual[2], second, 1e-6, "KL second example")
  near(actual[3], 0, 0, "KL zero labels")
  near(values[2]:totable(), (first + second) / 3, 1e-6, "KL mean")
  near(values[3]:totable(), first + second, 1e-6, "KL sum")
  assert_true(tfa.losses.KLDivergence == tfa.losses.kl_divergence)
  assert_true(tfa.losses.kl_divergence_loss == tfa.losses.kl_divergence)
  session:close()
end)

test("precision matches binary and multiclass counts", function()
  local graph = tf.Graph.new()
  local labels = tf.placeholder(graph, "float32", { 4 }, "precision_labels")
  local predictions = tf.placeholder(graph, "float32", { 4 },
    "precision_predictions")
  local classes = tf.placeholder(graph, "int32", { 5 },
    "precision_classes")
  local class_labels = tf.placeholder(graph, "float32", { 5, 3 },
    "precision_class_labels")
  local outputs = {
    tfa.metrics.precision(graph, predictions, labels,
      { threshold = 0.5, name = "precision_binary" }),
    tfa.metrics.precision(graph, predictions, labels,
      { threshold = 0.75, name = "precision_threshold" }),
    tfa.metrics.precision(graph, classes, class_labels,
      { average = "macro", name = "precision_macro" }),
    tfa.metrics.precision(graph, classes, class_labels,
      { average = "micro", name = "precision_micro" }),
    tfa.metrics.precision(graph, classes, class_labels,
      { average = "weighted", name = "precision_weighted" }),
  }
  local session = tf.Session.new(graph)
  local values = session:run({
    [labels] = { 1, 1, 0, 0 },
    [predictions] = { 0.9, 0.7, 0.8, 0.1 },
    [classes] = { 0, 0, 1, 2, 2 },
    [class_labels] = { { 1, 0, 0 }, { 1, 0, 0 }, { 1, 0, 0 },
      { 0, 1, 0 }, { 0, 0, 1 } },
  }, outputs)
  near(values[1]:totable(), 2 / 3, 1e-6, "binary precision")
  near(values[2]:totable(), 1 / 2, 1e-6, "thresholded precision")
  near(values[3]:totable(), 1 / 2, 1e-6, "macro precision")
  near(values[4]:totable(), 3 / 5, 1e-6, "micro precision")
  near(values[5]:totable(), 7 / 10, 1e-6, "weighted precision")
  assert_true(tfa.metrics.Precision == tfa.metrics.precision)
  session:close()
end)

test("recall matches binary and multiclass counts", function()
  local graph = tf.Graph.new()
  local labels = tf.placeholder(graph, "float32", { 4 }, "recall_labels")
  local predictions = tf.placeholder(graph, "float32", { 4 },
    "recall_predictions")
  local classes = tf.placeholder(graph, "int32", { 5 }, "recall_classes")
  local class_labels = tf.placeholder(graph, "float32", { 5, 3 },
    "recall_class_labels")
  local outputs = {
    tfa.metrics.recall(graph, predictions, labels,
      { threshold = 0.5, name = "recall_binary" }),
    tfa.metrics.recall(graph, predictions, labels,
      { threshold = 0.75, name = "recall_threshold" }),
    tfa.metrics.recall(graph, classes, class_labels,
      { average = "macro", name = "recall_macro" }),
    tfa.metrics.recall(graph, classes, class_labels,
      { average = "micro", name = "recall_micro" }),
    tfa.metrics.recall(graph, classes, class_labels,
      { average = "weighted", name = "recall_weighted" }),
  }
  local session = tf.Session.new(graph)
  local values = session:run({
    [labels] = { 1, 1, 0, 0 },
    [predictions] = { 0.9, 0.7, 0.8, 0.1 },
    [classes] = { 0, 0, 1, 2, 2 },
    [class_labels] = { { 1, 0, 0 }, { 1, 0, 0 }, { 1, 0, 0 },
      { 0, 1, 0 }, { 0, 0, 1 } },
  }, outputs)
  near(values[1]:totable(), 1, 1e-6, "binary recall")
  near(values[2]:totable(), 1 / 2, 1e-6, "thresholded recall")
  near(values[3]:totable(), 5 / 9, 1e-6, "macro recall")
  near(values[4]:totable(), 3 / 5, 1e-6, "micro recall")
  near(values[5]:totable(), 3 / 5, 1e-6, "weighted recall")
  assert_true(tfa.metrics.Recall == tfa.metrics.recall)
  session:close()
end)

test("sobel_edges matches reflected depthwise Sobel gradients", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 1, 3, 3, 2 },
    "sobel_input")
  local edges = tfa.image.sobel_edges(graph, input, { name = "sobel" })
  local session = tf.Session.new(graph)
  local tensor = session:run({ [input] = { {
    { { 1, 3 }, { 2, 3 }, { 3, 3 } },
    { { 4, 3 }, { 5, 3 }, { 6, 3 } },
    { { 7, 3 }, { 8, 3 }, { 9, 3 } },
  } } }, { edges })[1]
  assert_eq(table.concat(tensor:shape(), ","), "1,3,3,2,2")
  local values = tensor:totable()
  near(values[1][2][2][1][1], 24, 1e-6, "sobel dy at center")
  near(values[1][2][2][1][2], 8, 1e-6, "sobel dx at center")
  for row = 1, 3 do
    for column = 1, 3 do
      near(values[1][row][column][2][1], 0, 0,
        "sobel constant dy")
      near(values[1][row][column][2][2], 0, 0,
        "sobel constant dx")
    end
  end
  session:close()
end)

test("AUC matches the Keras threshold sweep and PR interpolation", function()
  local graph = tf.Graph.new()
  local labels = tf.placeholder(graph, "float32", { 4 }, "auc_labels")
  local predictions = tf.placeholder(graph, "float32", { 4 },
    "auc_predictions")
  local perfect_labels = tf.placeholder(graph, "float32", { 4 },
    "auc_perfect_labels")
  local perfect_predictions = tf.placeholder(graph, "float32", { 4 },
    "auc_perfect_predictions")
  local logits = tf.placeholder(graph, "float32", { 2 }, "auc_logit_inputs")
  local logit_labels = tf.placeholder(graph, "float32", { 2 },
    "auc_logit_labels")
  local roc = tfa.metrics.auc(graph, predictions, labels,
    { num_thresholds = 3, name = "auc_roc" })
  local pr = tfa.metrics.auc(graph, perfect_predictions, perfect_labels,
    { num_thresholds = 3, curve = "PR", name = "auc_pr" })
  local logit_roc = tfa.metrics.auc(graph, logits, logit_labels,
    { num_thresholds = 3, from_logits = true, name = "auc_logits" })
  local minor = tfa.metrics.auc(graph, predictions, labels,
    { num_thresholds = 3, summation_method = "minoring", name = "auc_minor" })
  local major = tfa.metrics.auc(graph, predictions, labels,
    { num_thresholds = 3, summation_method = "majoring", name = "auc_major" })
  local session = tf.Session.new(graph)
  local values = session:run({
    [labels] = { 0, 0, 1, 1 },
    [predictions] = { 0, 0.5, 0.3, 0.9 },
    [perfect_labels] = { 0, 0, 1, 1 },
    [perfect_predictions] = { 0.1, 0.2, 0.8, 0.9 },
    [logits] = { -10, 10 },
    [logit_labels] = { 0, 1 },
  }, { roc, pr, logit_roc, minor, major })
  near(values[1]:totable(), 0.75, 1e-6, "ROC AUC")
  near(values[2]:totable(), 1, 1e-6, "PR AUC")
  near(values[3]:totable(), 1, 1e-6, "logit ROC AUC")
  near(values[4]:totable(), 0.5, 1e-6, "minor ROC AUC")
  near(values[5]:totable(), 1, 1e-6, "major ROC AUC")
  session:close()
end)

test("sensitivity and specificity honor their operating constraints", function()
  local graph = tf.Graph.new()
  local labels = tf.placeholder(graph, "float32", { 5 }, "operating_labels")
  local predictions = tf.placeholder(graph, "float32", { 5 },
    "operating_predictions")
  local sensitivity = tfa.metrics.sensitivity_at_specificity(graph,
    predictions, labels, 0.5, { name = "sensitivity_at_half" })
  local specificity = tfa.metrics.specificity_at_sensitivity(graph,
    predictions, labels, { sensitivity = 0.5, name = "specificity_at_half" })
  local session = tf.Session.new(graph)
  local values = session:run({
    [labels] = { 0, 0, 0, 1, 1 },
    [predictions] = { 0, 0.3, 0.8, 0.3, 0.8 },
  }, { sensitivity, specificity })
  near(values[1]:totable(), 0.5, 1e-6, "sensitivity at 0.5 specificity")
  near(values[2]:totable(), 2 / 3, 1e-6, "specificity at 0.5 sensitivity")
  assert_true(tfa.metrics.SensitivityAtSpecificity ==
    tfa.metrics.sensitivity_at_specificity)
  assert_true(tfa.metrics.SpecificityAtSensitivity ==
    tfa.metrics.specificity_at_sensitivity)
  session:close()
end)

test("noisy dense applies factorised noise to its mean weights", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 2, 2 }, "noisy_input")
  local noise_kernel = tf.placeholder(graph, "float32", { 2, 3 },
    "noise_kernel")
  local noise_bias = tf.placeholder(graph, "float32", { 3 }, "noise_bias")
  local layer = tfa.layers.noisy_dense(graph, input, 3,
    { input_shape = { 2, 2 }, noise_kernel = noise_kernel,
      noise_bias = noise_bias, name = "noisy" })
  local session = tf.Session.new(graph)
  session:run({}, {}, { unpack(layer.initializers) })
  local values = session:run({
    [input] = { { 1, 2 }, { 3, 4 } },
    [noise_kernel] = { { 1, 0, 0 }, { 0, 1, 0 } },
    [noise_bias] = { 1, 2, 3 },
  }, { layer.output, layer.kernel, layer.bias })
  local output = values[1]:totable()
  local kernel = values[2]:totable()
  local bias = values[3]:totable()
  local rows = { { 1, 2 }, { 3, 4 } }
  for row = 1, 2 do
    for unit = 1, 3 do
      local expected = bias[unit]
      for feature = 1, 2 do
        expected = expected + rows[row][feature] * kernel[feature][unit]
      end
      near(output[row][unit], expected, 1e-6,
        "noisy dense output " .. row .. "," .. unit)
    end
  end
  assert_eq(#layer.variables, 4)
  assert_eq(#layer.initializers, 4)
  assert_true(tfa.layers.NoisyDense == tfa.layers.noisy_dense)
  session:close()
end)

test("noisy dense supports independent noise and disabled noise", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 2, 2 }, "noisy_variants")
  local removed = tfa.layers.noisy_dense(graph, input, 3,
    { input_shape = { 2, 2 }, remove_noise = true,
      name = "noisy_removed" })
  local independent = tfa.layers.noisy_dense(graph, input, 3,
    { input_shape = { 2, 2 }, use_factorised = false, activation = "relu",
      seed = 7, name = "noisy_independent" })
  local no_bias = tfa.layers.noisy_dense(graph, input, 3,
    { input_shape = { 2, 2 }, use_bias = false, remove_noise = true,
      name = "noisy_no_bias" })
  local initializers = {}
  for _, set in ipairs({ removed.initializers, independent.initializers,
      no_bias.initializers }) do
    for _, initializer in ipairs(set) do
      initializers[#initializers + 1] = initializer
    end
  end
  local session = tf.Session.new(graph)
  session:run({}, {}, initializers)
  local values = session:run({ [input] = { { 1, -2 }, { 3, 4 } } },
    { removed.kernel, removed.mu_kernel.read, independent.output,
      independent.kernel, independent.bias })
  local kernel = values[1]:totable()
  local mean = values[2]:totable()
  for feature = 1, 2 do
    for unit = 1, 3 do
      near(kernel[feature][unit], mean[feature][unit], 0,
        "removed-noise kernel")
    end
  end
  assert_eq(table.concat(values[3]:shape(), ","), "2,3")
  assert_eq(no_bias.bias, nil)
  assert_eq(#no_bias.initializers, 2)
  session:close()
end)

test("translate shifts images and fills outside pixels", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 1, 3, 3, 1 },
    "translate_input")
  local shifted = tfa.image.translate(graph, input, { 1, 0 },
    { fill_value = 9, name = "translate_right" })
  local session = tf.Session.new(graph)
  local tensor = session:run({ [input] = { {
    { { 1 }, { 2 }, { 3 } },
    { { 4 }, { 5 }, { 6 } },
    { { 7 }, { 8 }, { 9 } },
  } } }, { shifted })[1]
  assert_eq(table.concat(tensor:shape(), ","), "1,3,3,1")
  local values = tensor:totable()[1]
  local expected = {
    { 9, 1, 2 },
    { 9, 4, 5 },
    { 9, 7, 8 },
  }
  for row = 1, 3 do
    for column = 1, 3 do
      near(values[row][column][1], expected[row][column], 0,
        "translated pixel " .. row .. "," .. column)
    end
  end
  assert_true(tfa.image.Translate == tfa.image.translate)
  session:close()
end)

test("translate accepts rank-3 images and per-image translations", function()
  local graph = tf.Graph.new()
  local image = tf.placeholder(graph, "float32", { 2, 2, 1 }, "translate_hwc")
  local batched = tf.placeholder(graph, "float32", { 2, 2, 2, 1 },
    "translate_batch")
  local moved = tfa.image.translate(graph, image, { 1, 0 },
    { rank = 3, interpolation = "nearest", name = "translate_rank3" })
  local batch_moved = tfa.image.translate(graph, batched,
    { { 1, 0 }, { 0, 1 } }, { name = "translate_batch" })
  local session = tf.Session.new(graph)
  local values = session:run({
    [image] = { { { 1 }, { 2 } }, { { 3 }, { 4 } } },
    [batched] = {
      { { { 1 }, { 2 } }, { { 3 }, { 4 } } },
      { { { 1 }, { 2 } }, { { 3 }, { 4 } } },
    },
  }, { moved, batch_moved })
  assert_eq(table.concat(values[1]:shape(), ","), "2,2,1")
  near(values[1]:totable()[1][2][1], 1, 0, "rank-3 translation")
  assert_eq(table.concat(values[2]:shape(), ","), "2,2,2,1")
  near(values[2]:totable()[1][1][1][1], 0, 0, "first batch fill")
  near(values[2]:totable()[2][2][1][1], 1, 0, "second batch shift down")
  session:close()
end)

test("precision and recall at operating points match Keras examples", function()
  local graph = tf.Graph.new()
  local recall_labels = tf.placeholder(graph, "float32", { 5 },
    "par_labels")
  local recall_predictions = tf.placeholder(graph, "float32", { 5 },
    "par_predictions")
  local precision_labels = tf.placeholder(graph, "float32", { 4 },
    "rap_labels")
  local precision_predictions = tf.placeholder(graph, "float32", { 4 },
    "rap_predictions")
  local precision = tfa.metrics.precision_at_recall(graph,
    recall_predictions, recall_labels, 0.5, { name = "par" })
  local recall = tfa.metrics.recall_at_precision(graph, precision_predictions,
    precision_labels, { precision = 0.8, name = "rap" })
  local session = tf.Session.new(graph)
  local values = session:run({
    [recall_labels] = { 0, 0, 0, 1, 1 },
    [recall_predictions] = { 0, 0.3, 0.8, 0.3, 0.8 },
    [precision_labels] = { 0, 0, 1, 1 },
    [precision_predictions] = { 0, 0.5, 0.3, 0.9 },
  }, { precision, recall })
  near(values[1]:totable(), 0.5, 1e-6, "precision at 0.5 recall")
  near(values[2]:totable(), 0.5, 1e-6, "recall at 0.8 precision")
  assert_true(tfa.metrics.PrecisionAtRecall ==
    tfa.metrics.precision_at_recall)
  assert_true(tfa.metrics.RecallAtPrecision ==
    tfa.metrics.recall_at_precision)
  session:close()
end)

test("ConditionalGradient applies the Frobenius-norm update", function()
  local graph = tf.Graph.new()
  local weight = tf.variable(graph, { { 1, 2 } }, "cg_weight",
    { dtype = "float32" })
  local gradient = tf.constant(graph, { { 0.1, 0.1 } }, "cg_gradient",
    "float32")
  local loss = tf.reduce_sum(graph, tf.mul(graph, weight.read, gradient,
    "cg_weighted", "float32"), { 0, 1 }, "cg_loss", "float32")
  local targets, gradients, state = tfa.optimizers.ConditionalGradient(graph,
    loss, { weight }, { learning_rate = 0.5, lambda_ = 0.01, name = "cg" })
  assert_eq(#targets, 1)
  assert_eq(#gradients, 1)
  assert_eq(#state.initializers, 0)
  local session = tf.Session.new(graph)
  session:run({}, {}, { weight.initializer })
  session:run({}, {}, targets)
  local values = session:run({}, { weight.read })[1]:totable()[1]
  local norm = math.sqrt(0.1 * 0.1 + 0.1 * 0.1)
  for index = 1, 2 do
    local original = index
    local expected = original * 0.5 - 0.5 * 0.01 * 0.1 / (norm + 1e-7)
    near(values[index], expected, 1e-6,
      "conditional gradient value " .. index)
  end
  session:close()
end)

test("random_hsv_in_yiq matches identity and saturation math", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 1, 2, 2, 3 },
    "hsv_input")
  local identity = tfa.image.random_hsv_in_yiq(graph, input,
    { name = "hsv_identity" })
  local grayscale = tfa.image.random_hsv_in_yiq(graph, input,
    { lower_saturation = 0, upper_saturation = 0, name = "hsv_gray" })
  local random = tfa.image.random_hsv_in_yiq(graph, input,
    { max_delta_hue = 0.25, lower_saturation = 0.8, upper_saturation = 1.2,
      lower_value = 0.8, upper_value = 1.2, seed = 11, name = "hsv_random" })
  local pixels = {
    { { 1, 0, 0 }, { 0, 1, 0 } },
    { { 0, 0, 1 }, { 1, 1, 1 } },
  }
  local session = tf.Session.new(graph)
  local values = session:run({ [input] = { pixels } },
    { identity, grayscale, random })
  local identity_values = values[1]:totable()[1]
  local gray_values = values[2]:totable()[1]
  for row = 1, 2 do
    for column = 1, 2 do
      local pixel = pixels[row][column]
      local luminance = 0.299 * pixel[1] + 0.587 * pixel[2] +
        0.114 * pixel[3]
      for channel = 1, 3 do
        near(identity_values[row][column][channel], pixel[channel], 1e-5,
          "HSV identity")
        near(gray_values[row][column][channel], luminance, 1e-5,
          "HSV saturation zero")
      end
    end
  end
  assert_eq(table.concat(values[3]:shape(), ","), "1,2,2,3")
  local random_values = values[3]:totable()[1]
  for row = 1, 2 do
    for column = 1, 2 do
      for channel = 1, 3 do
        local value = random_values[row][column][channel]
        assert_true(value == value and value ~= math.huge and
          value ~= -math.huge, "random HSV should stay finite")
      end
    end
  end
  assert_true(tfa.image.RandomHSVInYIQ == tfa.image.random_hsv_in_yiq)
  session:close()
end)

test("rotate matches identity and quarter-turn transforms", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 1, 2, 2, 1 },
    "rotate_input")
  local batch = tf.placeholder(graph, "float32", { 2, 2, 2, 1 },
    "rotate_batch")
  local identity = tfa.image.rotate(graph, input, 0, { name = "rotate_zero" })
  local quarter = tfa.image.rotate(graph, input, math.pi / 2,
    { name = "rotate_quarter" })
  local per_image = tfa.image.rotate(graph, batch, { 0, math.pi / 2 },
    { name = "rotate_batch" })
  local session = tf.Session.new(graph)
  local values = session:run({
    [input] = { { { { 1 }, { 2 } }, { { 3 }, { 4 } } } },
    [batch] = {
      { { { 1 }, { 2 } }, { { 3 }, { 4 } } },
      { { { 1 }, { 2 } }, { { 3 }, { 4 } } },
    },
  }, { identity, quarter, per_image })
  local expected_identity = { { 1, 2 }, { 3, 4 } }
  local expected_quarter = { { 2, 4 }, { 1, 3 } }
  for row = 1, 2 do
    for column = 1, 2 do
      near(values[1]:totable()[1][row][column][1],
        expected_identity[row][column], 0, "identity rotation")
      near(values[2]:totable()[1][row][column][1],
        expected_quarter[row][column], 0, "quarter rotation")
      near(values[3]:totable()[2][row][column][1],
        expected_quarter[row][column], 0, "batch quarter rotation")
    end
  end
  assert_true(tfa.image.Rotate == tfa.image.rotate)
  session:close()
end)

test("Poincare normalization projects vectors into the unit ball", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 3, 2 }, "poincare_input")
  local layer = tfa.layers.poincare_normalize(graph, input,
    { axis = 1, epsilon = 1e-5, name = "poincare" })
  local session = tf.Session.new(graph)
  local values = session:run({ [input] = {
    { 3, 4 },
    { 0.1, 0.2 },
    { 0, 0 },
  } }, { layer.output })[1]:totable()
  local boundary = 1 - 1e-5
  near(values[1][1], boundary * 3 / 5, 1e-6, "Poincare first coordinate")
  near(values[1][2], boundary * 4 / 5, 1e-6, "Poincare second coordinate")
  near(values[2][1], 0.1, 1e-6, "Poincare inside ball first")
  near(values[2][2], 0.2, 1e-6, "Poincare inside ball second")
  near(values[3][1], 0, 0, "Poincare zero vector first")
  near(values[3][2], 0, 0, "Poincare zero vector second")
  assert_true(tfa.layers.PoincareNormalize ==
    tfa.layers.poincare_normalize)
  session:close()
end)

test("bilinear_resize interpolates rank-4 images", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 1, 4, 4, 1 },
    "bilinear_input")
  local resized = tfa.image.bilinear_resize(graph, input, 2, 2,
    "bilinear_resize", { half_pixel_centers = false })
  local session = tf.Session.new(graph)
  local values = session:run({ [input] = { {
    { { 1 }, { 2 }, { 3 }, { 4 } },
    { { 5 }, { 6 }, { 7 }, { 8 } },
    { { 9 }, { 10 }, { 11 }, { 12 } },
    { { 13 }, { 14 }, { 15 }, { 16 } },
  } } }, { resized })[1]:totable()[1]
  assert_eq(table.concat({ #values, #values[1], #values[1][1] }, ","),
    "2,2,1")
  near(values[1][1][1], 1, 0, "bilinear top-left")
  near(values[1][2][1], 3, 0, "bilinear top-right")
  near(values[2][1][1], 9, 0, "bilinear bottom-left")
  near(values[2][2][1], 11, 0, "bilinear bottom-right")
  assert_true(tfa.image.BilinearResize == tfa.image.bilinear_resize)
  session:close()
end)

test("area_resize averages source pixels", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 1, 4, 4, 1 },
    "area_input")
  local resized = tfa.image.area_resize(graph, input, 2, 2, "area_resize")
  local session = tf.Session.new(graph)
  local values = session:run({ [input] = { {
    { { 1 }, { 2 }, { 3 }, { 4 } },
    { { 5 }, { 6 }, { 7 }, { 8 } },
    { { 9 }, { 10 }, { 11 }, { 12 } },
    { { 13 }, { 14 }, { 15 }, { 16 } },
  } } }, { resized })[1]:totable()[1]
  assert_eq(table.concat({ #values, #values[1], #values[1][1] }, ","),
    "2,2,1")
  near(values[1][1][1], 3.5, 1e-6, "area top-left")
  near(values[1][2][1], 5.5, 1e-6, "area top-right")
  near(values[2][1][1], 11.5, 1e-6, "area bottom-left")
  near(values[2][2][1], 13.5, 1e-6, "area bottom-right")
  assert_true(tfa.image.AreaResize == tfa.image.area_resize)
  session:close()
end)

test("adjust_saturation scales HSV saturation", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 1, 1, 2, 3 },
    "saturation_input")
  local grayscale = tfa.image.adjust_saturation(graph, input, 0, "sat_zero")
  local identity = tfa.image.adjust_saturation(graph, input, 1, "sat_one")
  local session = tf.Session.new(graph)
  local values = session:run({ [input] = { { {
    { 1, 0, 0 }, { 0.2, 0.4, 0.8 },
  } } } }, { grayscale, identity })
  local gray = values[1]:totable()[1][1]
  local original = values[2]:totable()[1][1]
  near(gray[1][1], 1, 1e-6, "red saturation zero")
  near(gray[1][2], 1, 1e-6, "red saturation zero green")
  near(gray[1][3], 1, 1e-6, "red saturation zero blue")
  near(gray[2][1], 0.8, 1e-6, "mixed saturation zero")
  near(gray[2][2], 0.8, 1e-6, "mixed saturation zero green")
  near(gray[2][3], 0.8, 1e-6, "mixed saturation zero blue")
  near(original[2][1], 0.2, 1e-6, "saturation one red")
  near(original[2][2], 0.4, 1e-6, "saturation one green")
  near(original[2][3], 0.8, 1e-6, "saturation one blue")
  assert_true(tfa.image.AdjustSaturation == tfa.image.adjust_saturation)
  session:close()
end)

test("MovingAverage tracks and assigns averaged weights", function()
  local graph = tf.Graph.new()
  local weight = tf.variable(graph, { { 1, 2 } }, "ma_weight",
    { dtype = "float32" })
  local gradient = tf.constant(graph, { { 0.1, 0.1 } }, "ma_gradient",
    "float32")
  local loss = tf.reduce_sum(graph, tf.mul(graph, weight.read, gradient,
    "ma_weighted", "float32"), { 0, 1 }, "ma_loss", "float32")
  local optimizer = tfa.optimizers.MovingAverage("gradient_descent",
    { learning_rate = 1, average_decay = 0.5, name = "moving_average" })
  local targets, gradients, state = optimizer(graph, loss, { weight })
  assert_eq(#targets, 3)
  assert_eq(#gradients, 1)
  assert_eq(#state.average_variables, 1)
  local session = tf.Session.new(graph)
  session:run({}, {}, { weight.initializer, unpack(state.initializers) })
  session:run({}, {}, targets)
  session:run({}, {}, targets)
  local fast = session:run({}, { weight.read })[1]:totable()[1]
  near(fast[1], 0.8, 1e-6, "moving average fast first")
  near(fast[2], 1.8, 1e-6, "moving average fast second")
  session:run({}, {}, state.assign_averages)
  local averaged = session:run({}, { weight.read })[1]:totable()[1]
  near(averaged[1], 0.875, 1e-6, "moving average assigned first")
  near(averaged[2], 1.875, 1e-6, "moving average assigned second")
  session:close()
end)

test("snake activation matches the periodic formula", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 3 }, "snake_input")
  local output = tfa.activations.snake(graph, input, 1, "snake")
  local session = tf.Session.new(graph)
  local values = session:run({ [input] = { -1, 0, 1 } }, { output })[1]
    :totable()
  near(values[1], -0.29192656, 1e-6, "snake negative")
  near(values[2], 0, 1e-6, "snake zero")
  near(values[3], 1.7080734, 1e-6, "snake positive")
  session:close()
end)

test("TLU applies learned thresholds and affine slopes", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 2, 3 }, "tlu_input")
  local layer = tfa.layers.tlu(graph, input,
    { input_shape = { 2, 3 }, tau = 0.5, affine = true, alpha = 0.25,
      name = "tlu" })
  local session = tf.Session.new(graph)
  session:run({}, {}, { unpack(layer.initializers) })
  local values = session:run({ [input] = {
    { -1, 0, 1 },
    { 2, -2, 0 },
  } }, { layer.output })[1]:totable()
  local expected = {
    { 0.25, 0.5, 1 },
    { 2, 0, 0.5 },
  }
  for row = 1, 2 do
    for column = 1, 3 do
      near(values[row][column], expected[row][column], 1e-6,
        "TLU " .. row .. "," .. column)
    end
  end
  assert_eq(#layer.variables, 2)
  assert_true(tfa.layers.TLU == tfa.layers.tlu)
  session:close()
end)

test("polynomial crossing builds the Deep & Cross interaction", function()
  local graph = tf.Graph.new()
  local x0 = tf.placeholder(graph, "float32", { 2, 2 }, "cross_x0")
  local x = tf.placeholder(graph, "float32", { 2, 2 }, "cross_x")
  local layer = tfa.layers.polynomial_crossing(graph, x0, x,
    { input_shape = { 2, 2 }, diag_scale = 0.5, name = "cross" })
  local session = tf.Session.new(graph)
  session:run({}, {}, { unpack(layer.initializers) })
  local values = session:run({
    [x0] = { { 1, 2 }, { 3, 4 } },
    [x] = { { 0.5, 1 }, { 1.5, 2 } },
  }, { layer.output, layer.kernel.read, layer.bias.read })
  local kernel = values[2]:totable()
  local bias = values[3]:totable()
  local original = { { 1, 2 }, { 3, 4 } }
  local previous = { { 0.5, 1 }, { 1.5, 2 } }
  local output = values[1]:totable()
  for row = 1, 2 do
    for column = 1, 2 do
      local product = 0
      for feature = 1, 2 do
        product = product + previous[row][feature] * kernel[feature][column]
      end
      product = product + 0.5 * previous[row][column]
      local expected = original[row][column] * product +
        previous[row][column] + bias[column]
      near(output[row][column], expected, 1e-6,
        "polynomial crossing " .. row .. "," .. column)
    end
  end
  assert_true(tfa.layers.PolynomialCrossing ==
    tfa.layers.polynomial_crossing)
  session:close()
end)

test("sharpness preserves identity and smooths with factor zero", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 1, 5, 5, 1 },
    "sharpness_input")
  local identity = tfa.image.sharpness(graph, input, 1, "sharp_identity")
  local smoothed = tfa.image.sharpness(graph, input, 0, "sharp_smooth")
  local session = tf.Session.new(graph)
  local values = session:run({ [input] = { {
    { { 0 }, { 0 }, { 0 }, { 0 }, { 0 } },
    { { 0 }, { 0 }, { 0 }, { 0 }, { 0 } },
    { { 0 }, { 0 }, { 1 }, { 0 }, { 0 } },
    { { 0 }, { 0 }, { 0 }, { 0 }, { 0 } },
    { { 0 }, { 0 }, { 0 }, { 0 }, { 0 } },
  } } }, { identity, smoothed })
  local identity_values = values[1]:totable()[1]
  local smooth_values = values[2]:totable()[1]
  near(identity_values[3][3][1], 1, 0, "sharpness identity center")
  near(identity_values[1][1][1], 0, 0, "sharpness identity border")
  near(smooth_values[3][3][1], 5 / 13, 1e-6, "sharpness smooth center")
  near(smooth_values[2][3][1], 1 / 13, 1e-6, "sharpness smooth edge")
  assert_eq(table.concat(values[2]:shape(), ","), "1,5,5,1")
  assert_true(tfa.image.Sharpness == tfa.image.sharpness)
  session:close()
end)

test("Yogi matches its first adaptive update", function()
  local graph = tf.Graph.new()
  local weight = tf.variable(graph, { { 1 } }, "yogi_weight",
    { dtype = "float32" })
  local gradient = tf.constant(graph, { { 0.1 } }, "yogi_gradient",
    "float32")
  local loss = tf.reduce_sum(graph, tf.mul(graph, weight.read, gradient,
    "yogi_weighted", "float32"), { 0, 1 }, "yogi_loss", "float32")
  local targets, gradients, state = tfa.optimizers.Yogi(graph, loss,
    { weight }, { learning_rate = 0.01, name = "yogi" })
  assert_eq(#gradients, 1)
  assert_eq(#state.variables, 4)
  local session = tf.Session.new(graph)
  session:run({}, {}, { weight.initializer, unpack(state.initializers) })
  session:run({}, {}, targets)
  local values = session:run({}, {
    weight.read, state.beta1_power.read, state.beta2_power.read,
    state.variables[3].read, state.variables[4].read,
  })
  local beta1, beta2, epsilon, learning_rate = 0.9, 0.999, 1e-3, 0.01
  local squared = 0.1 * 0.1
  local accumulator = 1e-6 + (1 - beta2) * squared
  local corrected_lr = learning_rate * math.sqrt(1 - beta2) / (1 - beta1)
  local per_coordinate = corrected_lr / (math.sqrt(accumulator) + epsilon)
  local expected = 1 - per_coordinate * ((1 - beta1) * 0.1)
  near(values[1]:totable()[1][1], expected, 1e-6, "Yogi weight")
  near(values[2]:totable(), beta1 * beta1, 1e-6, "Yogi beta1 power")
  near(values[3]:totable(), beta2 * beta2, 1e-6, "Yogi beta2 power")
  near(values[4]:totable()[1][1], accumulator, 1e-9, "Yogi accumulator")
  near(values[5]:totable()[1][1], (1 - beta1) * 0.1, 1e-8, "Yogi momentum")
  session:close()
end)

test("transform applies projective rows and output shapes", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 1, 2, 2, 1 },
    "transform_input")
  local identity = tfa.image.transform(graph, input,
    { 1, 0, 0, 0, 1, 0, 0, 0 }, { name = "transform_identity" })
  local shifted = tfa.image.transform(graph, input,
    { 1, 0, -1, 0, 1, 0, 0, 0 }, { name = "transform_shift" })
  local resized = tfa.image.transform(graph, input,
    { 1, 0, 0, 0, 1, 0, 0, 0 },
    { name = "transform_resize", output_shape = { 3, 3 } })
  local session = tf.Session.new(graph)
  local values = session:run({ [input] = { { { { 1 }, { 2 } },
    { { 3 }, { 4 } } } } }, { identity, shifted, resized })
  near(values[1]:totable()[1][1][1][1], 1, 0, "transform identity")
  near(values[2]:totable()[1][1][1][1], 0, 0, "transform shift fill")
  near(values[2]:totable()[1][1][2][1], 1, 0, "transform shift value")
  assert_eq(table.concat(values[3]:shape(), ","), "1,3,3,1")
  assert_true(tfa.image.Transform == tfa.image.transform)
  session:close()
end)

test("cutout masks a requested rectangle", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 1, 3, 3, 1 },
    "cutout_input")
  local output = tfa.image.cutout(graph, input, { 2, 2 }, { 1, 1 },
    { name = "cutout", constant_values = 9 })
  local values = run(graph, { [input] = { {
    { { 1 }, { 2 }, { 3 } },
    { { 4 }, { 5 }, { 6 } },
    { { 7 }, { 8 }, { 9 } },
  } } }, output)[1]
  local expected = {
    { 9, 9, 3 },
    { 9, 9, 6 },
    { 7, 8, 9 },
  }
  for row = 1, 3 do
    for column = 1, 3 do
      near(values[row][column][1], expected[row][column], 0,
        "cutout " .. row .. "," .. column)
    end
  end
  assert_true(tfa.image.Cutout == tfa.image.cutout)
end)

test("random_cutout masks each batch image", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 2, 4, 4, 1 },
    "random_cutout_input")
  local output = tfa.image.random_cutout(graph, input, { 2, 2 },
    { name = "random_cutout", constant_values = 0, seed = 7 })
  local session = tf.Session.new(graph)
  local values = session:run({ [input] = {
    { { { 1 }, { 1 }, { 1 }, { 1 } }, { { 1 }, { 1 }, { 1 }, { 1 } },
      { { 1 }, { 1 }, { 1 }, { 1 } }, { { 1 }, { 1 }, { 1 }, { 1 } } },
    { { { 1 }, { 1 }, { 1 }, { 1 } }, { { 1 }, { 1 }, { 1 }, { 1 } },
      { { 1 }, { 1 }, { 1 }, { 1 } }, { { 1 }, { 1 }, { 1 }, { 1 } } },
  } }, { output })[1]
  assert_eq(table.concat(values:shape(), ","), "2,4,4,1")
  local zeros = 0
  for batch = 1, 2 do
    for row = 1, 4 do
      for column = 1, 4 do
        if values:totable()[batch][row][column][1] == 0 then
          zeros = zeros + 1
        end
      end
    end
  end
  assert_eq(zeros, 8)
  assert_true(tfa.image.RandomCutout == tfa.image.random_cutout)
  session:close()
end)

test("adjust_hue rotates normalized hue values", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 1, 1, 1, 3 },
    "hue_input")
  local identity = tfa.image.adjust_hue(graph, input, 0, "hue_identity")
  local rotated = tfa.image.adjust_hue(graph, input, 1 / 3, "hue_rotated")
  local session = tf.Session.new(graph)
  local values = session:run({ [input] = { { { { 1, 0, 0 } } } } },
    { identity, rotated })
  local original = values[1]:totable()[1][1][1]
  local changed = values[2]:totable()[1][1][1]
  near(original[1], 1, 0, "hue identity red")
  near(original[2], 0, 0, "hue identity green")
  near(changed[1], 0, 1e-6, "hue rotation red")
  near(changed[2], 1, 1e-6, "hue rotation green")
  near(changed[3], 0, 1e-6, "hue rotation blue")
  assert_true(tfa.image.AdjustHue == tfa.image.adjust_hue)
  session:close()
end)

test("sparsemax produces sparse probability rows", function()
  local graph = tf.Graph.new()
  local input = tf.placeholder(graph, "float32", { 2, 3 }, "sparsemax_input")
  local layer = tfa.layers.sparsemax(graph, input,
    { input_shape = { 2, 3 }, name = "sparsemax" })
  local session = tf.Session.new(graph)
  local values = session:run({ [input] = {
    { -1, 0, 1 },
    { -5, 1, 2 },
  } }, { layer.output })[1]:totable()
  local expected = {
    { 0, 0, 1 },
    { 0, 0, 1 },
  }
  for row = 1, 2 do
    local total = 0
    for column = 1, 3 do
      near(values[row][column], expected[row][column], 1e-6,
        "sparsemax " .. row .. "," .. column)
      total = total + values[row][column]
    end
    near(total, 1, 1e-6, "sparsemax row sum")
  end
  assert_true(tfa.layers.Sparsemax == tfa.layers.sparsemax)
  session:close()
end)

test("new parity options reject invalid values", function()
  local graph = tf.Graph.new()
  local labels = tf.placeholder(graph, "float32", { 1 }, "newparity_labels")
  local predictions = tf.placeholder(graph, "float32", { 1 },
    "newparity_predictions")
  local input = tf.placeholder(graph, "float32", { 1, 3, 3, 1 },
    "newparity_input")
  local invalid_tau = pcall(tfa.losses.pinball, graph, labels, predictions,
    { tau = 1.5 })
  assert_true(not invalid_tau, "pinball should reject tau outside [0, 1]")
  local invalid_gamma, gamma_error = pcall(
    tfa.losses.sigmoid_focal_crossentropy, graph, labels, predictions,
    { gamma = -1 })
  assert_true(not invalid_gamma,
    "sigmoid focal should reject a negative gamma")
  assert_true(tostring(gamma_error):find("gamma", 1, true) ~= nil)
  local invalid_padding = pcall(tfa.image.median_filter2d, graph, input,
    { padding = "WRAP" })
  assert_true(not invalid_padding,
    "median_filter2d should reject unsupported padding")
  local invalid_depth = pcall(tfa.layers.stochastic_depth, graph, input, input,
    { survival_probability = 2 })
  assert_true(not invalid_depth,
    "stochastic depth should reject survival_probability outside [0, 1]")
  local embeddings = tf.placeholder(graph, "float32", { 1, 2 },
    "newparity_embeddings")
  local invalid_distance = pcall(tfa.losses.triplet_semihard_loss, graph,
    labels, embeddings, { distance_metric = "cosine" })
  assert_true(not invalid_distance,
    "triplet semihard should reject an unknown distance metric")
  local invalid_mode = pcall(tfa.metrics.hamming_loss, graph, predictions,
    labels, { mode = "binary" })
  assert_true(not invalid_mode, "hamming loss should reject an unknown mode")
  local invalid_axis = pcall(tfa.layers.filter_response_normalization, graph,
    input, { input_shape = { 1, 3, 3, 1 }, axis = { 1, 3 } })
  assert_true(not invalid_axis,
    "filter response normalization should reject non-spatial axes")
  local invalid_mean_padding = pcall(tfa.image.mean_filter2d, graph, input,
    { padding = "WRAP" })
  assert_true(not invalid_mean_padding,
    "mean_filter2d should reject unsupported padding")
  local invalid_margin = pcall(tfa.losses.hinge, graph, labels, predictions,
    { margin = -1 })
  assert_true(not invalid_margin, "hinge should reject a negative margin")
  local invalid_epsilon = pcall(tfa.losses.kl_divergence, graph, labels,
    predictions, { epsilon = 0 })
  assert_true(not invalid_epsilon,
    "KL divergence should reject a non-positive epsilon")
  local invalid_precision = pcall(tfa.metrics.precision, graph, predictions,
    labels, { average = "median" })
  assert_true(not invalid_precision,
    "precision should reject an unknown average")
  local invalid_recall = pcall(tfa.metrics.recall, graph, predictions, labels,
    { average = "median" })
  assert_true(not invalid_recall, "recall should reject an unknown average")
  local invalid_sobel_dtype = pcall(tfa.image.sobel_edges, graph, input,
    { dtype = "int32" })
  assert_true(not invalid_sobel_dtype,
    "sobel_edges should reject a non-floating point dtype")
  local invalid_sobel_rank = pcall(tfa.image.sobel_edges, graph, input,
    { input_shape = { 3, 3, 1 } })
  assert_true(not invalid_sobel_rank,
    "sobel_edges should reject a non-rank-4 input shape")
  local invalid_curve = pcall(tfa.metrics.auc, graph, predictions, labels,
    { curve = "lift" })
  assert_true(not invalid_curve, "AUC should reject an unknown curve")
  local invalid_auc_thresholds = pcall(tfa.metrics.auc, graph, predictions,
    labels, { num_thresholds = 1 })
  assert_true(not invalid_auc_thresholds,
    "AUC should require more than one threshold")
  local invalid_sensitivity = pcall(tfa.metrics.sensitivity_at_specificity,
    graph, predictions, labels, 1.5)
  assert_true(not invalid_sensitivity,
    "sensitivity_at_specificity should reject targets outside [0, 1]")
  local invalid_specificity = pcall(tfa.metrics.specificity_at_sensitivity,
    graph, predictions, labels, -0.1)
  assert_true(not invalid_specificity,
    "specificity_at_sensitivity should reject targets outside [0, 1]")
  local invalid_units = pcall(tfa.layers.noisy_dense, graph, input, 0,
    { input_shape = { 1, 3, 3, 1 } })
  assert_true(not invalid_units,
    "noisy_dense should reject a non-positive unit count")
  local invalid_noise = pcall(tfa.layers.noisy_dense, graph, input, 2,
    { input_shape = { 1, 3, 3, 1 }, noise_kernel = input })
  assert_true(not invalid_noise,
    "noisy_dense should require both kernel and bias noise")
  local invalid_translate = pcall(tfa.image.translate, graph, input, { 1, 0 },
    { interpolation = "cubic" })
  assert_true(not invalid_translate,
    "translate should reject unsupported interpolation")
  local invalid_translate_rank = pcall(tfa.image.translate, graph, input,
    { 1, 0 }, { rank = 5 })
  assert_true(not invalid_translate_rank,
    "translate should reject unsupported image ranks")
  local invalid_par = pcall(tfa.metrics.precision_at_recall, graph,
    predictions, labels, 1.5)
  assert_true(not invalid_par,
    "precision_at_recall should reject targets outside [0, 1]")
  local invalid_rap = pcall(tfa.metrics.recall_at_precision, graph,
    predictions, labels, -0.1)
  assert_true(not invalid_rap,
    "recall_at_precision should reject targets outside [0, 1]")
  local invalid_ord = pcall(tfa.optimizers.ConditionalGradient, graph, nil,
    { { ref = input } }, { ord = "l2" })
  assert_true(not invalid_ord,
    "ConditionalGradient should reject unsupported norms")
  local invalid_hsv_shape = pcall(tfa.image.random_hsv_in_yiq, graph, input,
    { input_shape = { 1, 3, 3, 1 } })
  assert_true(not invalid_hsv_shape,
    "random_hsv_in_yiq should require three color channels")
  local invalid_hsv_range = pcall(tfa.image.random_hsv_in_yiq, graph, input,
    { input_shape = { 1, 3, 3, 3 }, lower_saturation = 2,
      upper_saturation = 1 })
  assert_true(not invalid_hsv_range,
    "random_hsv_in_yiq should reject inverted saturation ranges")
  local invalid_angle = pcall(tfa.image.rotate, graph, input, { "bad" })
  assert_true(not invalid_angle,
    "rotate should reject non-numeric angles")
  local invalid_poincare = pcall(tfa.layers.poincare_normalize, graph, input,
    { epsilon = 1 })
  assert_true(not invalid_poincare,
    "poincare_normalize should reject epsilon outside (0, 1)")
  local invalid_bilinear = pcall(tfa.image.bilinear_resize, graph, input, 0,
    2)
  assert_true(not invalid_bilinear,
    "bilinear_resize should reject non-positive sizes")
  local invalid_area = pcall(tfa.image.area_resize, graph, input, 2, 2,
    "invalid_area", { dtype = "int32" })
  assert_true(not invalid_area,
    "area_resize should reject integer images")
  local invalid_saturation = pcall(tfa.image.adjust_saturation, graph, input,
    -1)
  assert_true(not invalid_saturation,
    "adjust_saturation should reject negative saturation")
  local invalid_decay = pcall(tfa.optimizers.MovingAverage,
    "gradient_descent", { average_decay = 1.5 })
  assert_true(not invalid_decay,
    "MovingAverage should reject average_decay outside [0, 1]")
  local invalid_start = pcall(tfa.optimizers.MovingAverage,
    "gradient_descent", { start_step = 1.5 })
  assert_true(not invalid_start,
    "MovingAverage should reject non-integer start steps")
  local invalid_frequency = pcall(tfa.activations.snake, graph, input, 0)
  assert_true(not invalid_frequency,
    "snake should reject non-positive frequencies")
  local invalid_tlu = pcall(tfa.layers.tlu, graph, input, {})
  assert_true(not invalid_tlu,
    "TLU should require an input shape")
  local invalid_projection = pcall(tfa.layers.polynomial_crossing, graph,
    input, input, { input_shape = { 1, 3, 3, 1 }, projection_dim = 5 })
  assert_true(not invalid_projection,
    "polynomial_crossing should reject an oversized projection")
  local invalid_sharpness = pcall(tfa.image.sharpness, graph, input, -1)
  assert_true(not invalid_sharpness,
    "sharpness should reject negative factors")
  local invalid_yogi_activation = pcall(tfa.optimizers.Yogi, graph, nil,
    { { ref = input } }, { activation = "relu" })
  assert_true(not invalid_yogi_activation,
    "Yogi should reject unsupported activation functions")
  local invalid_yogi_beta = pcall(tfa.optimizers.Yogi, graph, nil,
    { { ref = input } }, { beta1 = 1 })
  assert_true(not invalid_yogi_beta,
    "Yogi should reject beta1 outside [0, 1)")
  local invalid_transform = pcall(tfa.image.transform, graph, input,
    { 1, 0, 0 })
  assert_true(not invalid_transform,
    "transform should reject rows without eight entries")
  local invalid_output_shape = pcall(tfa.image.transform, graph, input,
    { 1, 0, 0, 0, 1, 0, 0, 0 }, { output_shape = { 0, 2 } })
  assert_true(not invalid_output_shape,
    "transform should reject non-positive output sizes")
  local invalid_cutout = pcall(tfa.image.cutout, graph, input, { 3, 2 },
    { 1, 1 })
  assert_true(not invalid_cutout,
    "cutout should reject odd mask dimensions")
  local invalid_random_cutout = pcall(tfa.image.random_cutout, graph, input,
    { 2, 2 }, { constant_values = math.huge })
  assert_true(not invalid_random_cutout,
    "random_cutout should reject non-finite fill values")
  local invalid_hue = pcall(tfa.image.adjust_hue, graph, input, math.huge)
  assert_true(not invalid_hue,
    "adjust_hue should reject non-finite deltas")
  local invalid_sparsemax = pcall(tfa.layers.sparsemax, graph, input,
    { input_shape = { 1, 3, 3, 1 }, axis = 8 })
  assert_true(not invalid_sparsemax,
    "sparsemax should reject out-of-range axes")
end)

io.write(string.format("\n%d addons tests, %d passed, %d failed\n",
  passed + failed, passed, failed))
if #failures > 0 then
  io.write("\n" .. table.concat(failures, "\n") .. "\n")
  os.exit(1)
end
