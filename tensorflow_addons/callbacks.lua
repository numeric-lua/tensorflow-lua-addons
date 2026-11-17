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

return Callbacks
