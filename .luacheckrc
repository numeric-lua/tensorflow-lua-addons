std = "luajit"
files["tests"] = {
  globals = { "test", "assert_true", "assert_eq", "assert_error", "TEST_ROOT" },
}
files["examples"] = {
  globals = { "assert" },
}
