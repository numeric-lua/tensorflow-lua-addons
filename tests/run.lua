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

io.write(string.format("\n%d addons tests, %d passed, %d failed\n",
  passed + failed, passed, failed))
if #failures > 0 then
  io.write("\n" .. table.concat(failures, "\n") .. "\n")
  os.exit(1)
end
