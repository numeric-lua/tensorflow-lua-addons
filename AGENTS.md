# AGENTS.md — tensorflow-lua-addons

Additional operators and training helpers layered on `tensorflow-lua`.

- Addons must use the public `tensorflow` API; do not modify the core binding
  from this repository.
- Every behavior change needs a graph-mode test with a real TensorFlow session.
- Prefer vectorized graph operations over Lua loops.
- Keep the API close to TensorFlow Addons naming and options.
- Target LuaJIT 2.1 and Lua 5.1-compatible syntax.
- Never add a runtime dependency on Python, Node, or PyTorch.
- Reuse `tensorflow-lua/third_party/libtensorflow` in local tests; do not vendor
  a second TensorFlow build.
