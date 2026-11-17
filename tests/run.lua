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

io.write(string.format("\n%d addons tests, %d passed, %d failed\n",
  passed + failed, passed, failed))
if #failures > 0 then
  io.write("\n" .. table.concat(failures, "\n") .. "\n")
  os.exit(1)
end
