local Callbacks = {}

function Callbacks.TimeStopping(seconds)
  if type(seconds) ~= "number" or seconds <= 0 then
    error("TimeStopping requires a positive number of seconds", 2)
  end
  local callback = { seconds = seconds, started = nil, stopped_epoch = nil }
  function callback.on_train_begin()
    callback.started = os.time()
    callback.stopped_epoch = nil
  end
  function callback.on_epoch_end(history, epoch)
    if callback.started and os.time() - callback.started >= callback.seconds then
      history.stop_training = true
      callback.stopped_epoch = epoch
    end
  end
  return callback
end

function Callbacks.TerminateOnNaN()
  local callback = {}
  function callback.on_epoch_end(history)
    for key, values in pairs(history) do
      if type(values) == "table" then
        for _, value in ipairs(values) do
          if type(value) == "number" and
              (value ~= value or value == math.huge or value == -math.huge) then
            history.stop_training = true
            callback.stopped_by = key
            return
          end
        end
      end
    end
  end
  return callback
end

function Callbacks.CSVLogger(path, options)
  options = options or {}
  if type(path) ~= "string" or path == "" then
    error("CSVLogger requires a path", 2)
  end
  local callback = { path = path, header = nil, options = options }
  function callback.on_train_begin(history)
    local keys = {}
    for key in pairs(history or {}) do
      if type(history[key]) == "table" then keys[#keys + 1] = key end
    end
    table.sort(keys)
    callback.header = keys
    local file = assert(io.open(path, "w"))
    file:write(table.concat(keys, ",") .. "\n")
    file:close()
  end
  function callback.on_epoch_end(history)
    if callback.header == nil then callback.on_train_begin(history) end
    local values = {}
    for index, key in ipairs(callback.header) do
      local entries = history[key] or {}
      values[index] = tostring(entries[#entries])
    end
    local file = assert(io.open(path, "a"))
    file:write(table.concat(values, ",") .. "\n")
    file:close()
  end
  return callback
end

function Callbacks.BackupAndRestore(path, options)
  if type(path) ~= "string" or path == "" then
    error("BackupAndRestore requires a path", 2)
  end
  local callback = { path = path, restored = false, saved = false,
    options = options or {} }
  function callback.on_train_begin(_, model)
    local ok = pcall(model.load_training, model, path)
    callback.restored = ok
  end
  function callback.on_epoch_end(_, _, model)
    model:save_training(path)
    callback.saved = true
  end
  return callback
end

-- Lightweight TQDM-style progress callback implemented with plain Lua I/O.
-- It mirrors tfa.callbacks.TQDMProgressBar's lifecycle without a Python or
-- external tqdm dependency.
function Callbacks.TQDMProgressBar(options)
  options = options or {}
  local callback = {
    metrics_separator = options.metrics_separator or " - ",
    metrics_format = options.metrics_format or "{name}: {value:.4f}",
    show_epoch_progress = options.show_epoch_progress ~= false,
    show_overall_progress = options.show_overall_progress ~= false,
    leave_epoch_progress = options.leave_epoch_progress ~= false,
    leave_overall_progress = options.leave_overall_progress ~= false,
    update_per_second = options.update_per_second or 10,
    last_message = nil,
  }
  if type(callback.update_per_second) ~= "number" or
      callback.update_per_second <= 0 then
    error("TQDMProgressBar update_per_second must be positive", 2)
  end
  function callback.on_train_begin(params)
    callback.params = params or {}
    callback.current_epoch = 0
    callback.steps_so_far = 0
    callback.started = os.time()
    if callback.show_overall_progress then
      io.write("Training for " .. tostring(callback.params.epochs or "?") ..
        " epochs\n")
    end
  end
  function callback.on_epoch_begin(epoch)
    callback.current_epoch = epoch
    callback.steps_so_far = 0
    callback.total_steps = callback.params and callback.params.steps or nil
    if callback.show_epoch_progress then
      io.write(string.format("Epoch %d/%s\n", epoch + 1,
        tostring(callback.params and callback.params.epochs or "?")))
    end
  end
  function callback.on_batch_end(batch, logs)
    callback.steps_so_far = batch + 1
    callback.last_logs = logs
  end
  function callback.on_epoch_end(epoch, logs)
    callback.current_epoch = epoch
    callback.last_logs = logs
    local parts = {}
    for key, value in pairs(logs or {}) do
      if type(value) == "number" then
        local formatted = callback.metrics_format:gsub("{name}", key)
        formatted = formatted:gsub("{value:[^}]+}", string.format("%.4f", value))
        formatted = formatted:gsub("{value}", tostring(value))
        parts[#parts + 1] = formatted
      end
    end
    table.sort(parts)
    local message = table.concat(parts, callback.metrics_separator)
    if callback.total_steps ~= nil then
      message = string.format("%d/%d", callback.steps_so_far,
        callback.total_steps) .. callback.metrics_separator .. message
    end
    callback.last_message = message
    if callback.show_epoch_progress then io.write(message .. "\n") end
  end
  function callback.on_train_end()
    callback.elapsed = os.time() - (callback.started or os.time())
    if callback.show_overall_progress then
      io.write(string.format("Training complete in %ds\n", callback.elapsed))
    end
  end
  return callback
end

return Callbacks
