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
