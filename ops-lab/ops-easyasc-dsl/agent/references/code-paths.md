# Code Paths

Use this file when you already know the topic and need to find the implementation path quickly.
Do not read the whole repository first.

## Device family mapping

Use this mapping before interpreting hardware-specific branches:
- `device_type == "950"` -> C310
- `device_type.startswith("b")` -> C220
- in `easyasc/resources/tensorutils.h`, `__DAV_C310__` corresponds to `950`
- in `easyasc/resources/tensorutils.h`, `__DAV_C220_CUBE__` corresponds to `b*`
- do not invert this mapping when reading support checks, helper selection, or generated code paths

## Common Questions -> Files

### Repository structure or contributor architecture
- top-level layout: `agent/references/repo-map.md`
- contributor architecture and subsystem ownership: `doc/11_architecture_for_contributors.md`

### Public DSL surface
- architecture-specific exports: `easyasc/a2.py`, `easyasc/a5.py`
- treat `a2` and `a5` as parallel architecture surfaces for different instruction sequences, not as compatibility layers of each other
- decorators: `easyasc/decorators.py`
- flow helpers: `easyasc/flowcontrol.py`
- syntax sugar / AST rewrites: `easyasc/pythonic.py`
- runtime entry and `OpExec(..., simulator=...)` backend selection: `easyasc/torchplugin.py`
  - `simulator=True` now defaults to the V2 runtime; use `simulator="legacy"` when the old simulator is explicitly required

### Kernel and micro execution wrappers
- kernel lifecycle / generation hooks: `easyasc/kernelbase/kernelbase.py`
- micro module lifecycle: `easyasc/micro/micromodule.py`

### Lowering pipeline
- main lowering entry: `easyasc/parser/asc.py`
- pruning and cleanup: `easyasc/parser/asc_pruning.py`
- autosync insertion: `easyasc/parser/asc_autosync.py`
- op-specific lowering: `easyasc/parser/asc_handlers/`

### Simulator path
- simulator orchestration: `easyasc/simulator/base.py`
- per-core runtime: `easyasc/simulator/core.py`
- cube execution: `easyasc/simulator/cube.py`
- vec execution: `easyasc/simulator/vec.py`
- trace export: `easyasc/simulator/trace.py`
- shared layout/runtime helpers: `easyasc/simulator/shared/`
- micro runtime: `easyasc/simulator/micro/runtime/`
- experimental/runtime-rebuild path: `easyasc/simulator_v2/`
- V2 runtime activation and core/lane planning: `easyasc/simulator_v2/runtime/execution_plan.py`
- V2 default legacy-instruction bridge for `OpExec(simulator="v2")`: `easyasc/simulator_v2/compat/kernel_bridge.py`
- V2 control-flow bridge for mixed-lane / looped kernels: `easyasc/simulator_v2/compat/control_flow_bridge.py`
- V2 lane-level control interpreter and local event execution: `easyasc/simulator_v2/runtime/control_actor.py`
  - dynamic `get_buf` slot selection for loop-carried DBuff/TBuff/QBuff access is bridged here and resolved at runtime here
  - `create_varlist`, `GetValueFrom`, and `SetValueTo` also route through this control-flow path, including dynamic `VarList` index resolution
- V2 threaded pipe-worker lifecycle and pipe-local state retention: `easyasc/simulator_v2/runtime/pipe_worker.py`, `easyasc/simulator_v2/ops/dispatch.py`
  - vec mask state such as `set_mask/reset_mask` persists here across sequential V-pipe tasks on the same lane
- V2 Chrome-trace export path: `easyasc/simulator_v2/trace/chrome.py`

### Stub emitters and shortcuts
- operation emitters: `easyasc/stub_functions/`
- high-level matmul helpers: `easyasc/shortcuts/matmul.py`

### Runtime types and helper objects
- tensors, vars, regs, enums, instruction models: `easyasc/utils/`

### Test organization and runnable demos
- automated test layout: `historical testcases/README.md` (removed from this skill bundle)
- focused regression assets: `historical testcases/fixtures/` (removed from this skill bundle)
- manual/demo runners: `agent/example/demo/a2/`, `agent/example/demo/a5/`

## Topic Lookup

### Autosync or event-order issues
1. `agent/references/constraints/autosync.md`
2. `easyasc/parser/asc_autosync.py`
3. `easyasc/simulator_v2/compat/control_flow_bridge.py` for control-flow kernels, where V2 first splits cube/vec streams and injects autosync events before building lane programs
4. `easyasc/simulator_v2/runtime/control_actor.py` for V2 same-lane event execution and threaded pipe completion at `event_set` boundaries
5. `easyasc/parser/asc_handlers/`
6. relevant simulator pipe/runtime files under `easyasc/simulator/`

### Tiling or capacity legality
1. `agent/references/constraints/tiling.md`
2. `easyasc/shortcuts/matmul.py`
3. relevant kernel example from `agent/references/examples/kernel-catalog.md`
4. `agent/scripts/estimate_matmul_datamove.py`

### Counter lifetime or buffer ownership
1. `agent/references/constraints/counters.md`
2. target kernel source in `agent/example/kernels/`
3. lowering/runtime files only if behavior is still unclear

### Precision or cast placement
1. `agent/references/constraints/precision.md`
2. `easyasc/utils/`
3. matching lowering handler under `easyasc/parser/asc_handlers/`
4. simulator runtime if the question is execution-specific

### Tail handling or partial-tile writes
1. `agent/references/constraints/tail-safety.md`
2. matching kernel example in `agent/example/kernels/`
3. parser/simulator files if codegen/runtime behavior is in doubt

### Parser-side dead-code or pruning behavior
1. `easyasc/parser/asc_pruning.py`
2. related tests under `historical automated tests/` (removed from this skill bundle)

### Shape binding ambiguity
1. `easyasc/torchplugin.py`
2. matching tests under `historical automated tests/` (removed from this skill bundle)
3. affected kernel runner in `agent/example/kernels/`

Concrete trigger: when two scalar Var parameters have the same value at runtime
(e.g. `S1 == S2`), the framework cannot distinguish which scalar maps to which
tensor dimension. Fix: use `shape_bindings=` in `OpExec(...)()` call, or ensure
test shapes use distinct values for potentially ambiguous parameters.

### a2-specific cube → vec path
1. `agent/references/constraints/a2-device.md` — missing features and data path
2. `agent/references/patterns/a2-cube-vec.md` — GM workspace bridge pattern
3. `agent/references/constraints/vec-reduction-a2.md` — cmax + brcb for row max
4. `agent/references/constraints/vec-stride.md` — continuous vs sliced vec ops
5. `agent/example/kernels/a2/flash_attn_score.py` — working reference kernel

### a2-specific pure vec elementwise / quantization kernels
1. `agent/references/constraints/a2-vec-kernel.md` — pure vec kernel structure, flag flow, reinterpret, exact rounding
2. `agent/references/constraints/mask.md` — current vec mask semantics when masking is truly needed
3. `agent/references/constraints/vec-stride.md` — only if wide/narrow row interaction appears
4. `agent/example/kernels/a2/to_hif8_torch.py` — working vec-only quantization reference

### a2-specific cube → vec → cube delayed pipeline
1. `agent/references/constraints/a2-device.md` — both bridge restrictions (`L0C -> GM -> UB` and `UB -> GM -> L1`)
2. `agent/references/patterns/a2-cube-vec-cube.md` — one-tile lookahead schedule, shared `L0C`, delayed consumer
3. `agent/references/constraints/vec-reduction-a2.md` — running row-max update in `[M,1]` scalar format
4. `agent/references/constraints/vec-stride.md` — sliced `sub` against narrow row-max broadcast
5. `agent/example/kernels/a2/flash_attn_score_pv.py` — complete reference kernel

### a2-specific cube → vec → cube → vec delayed numerator accumulation
1. `agent/references/constraints/a2-device.md` — all three bridge restrictions plus scalar-state copy warning
2. `agent/references/patterns/a2-cube-vec-cube-vec.md` — one-tile lookahead with delayed final vec accumulation
3. `agent/references/constraints/vec-reduction-a2.md` — running max + delayed `expdiff` in `[M,1]` scalar format
4. `agent/references/constraints/vec-stride.md` — sliced scaling of `[M,128]` accumulators by narrow `[M,8]` broadcasts
5. `agent/example/kernels/a2/flash_attn_unnorm.py` — complete reference kernel with explicit local-event handoff for the final `accum_ub` store/reuse edge

### a2-specific cube → vec → cube → vec normalized online softmax
1. `agent/references/constraints/a2-device.md` — same triple-bridge hardware limits plus scalar-state copy warning
2. `agent/references/patterns/a2-cube-vec-cube-vec-softmax.md` — one-tile lookahead with running `row_max`, running `row_sum`, delayed `expdiff`, and final divide
3. `agent/references/constraints/online-softmax-tail.md` — score-domain `-inf` masking rule for non-aligned `S2` tails
4. `agent/references/constraints/reduction.md` — online softmax update order and "sum before cast" rule
5. `agent/references/constraints/vec-reduction-a2.md` — `cadd`/`brcb` row-sum pattern and final sliced `div`
6. `agent/references/constraints/vec-stride.md` — sliced division of `[M,128]` accumulators by `[M,8]` broadcasts
7. `agent/example/kernels/a2/flash_attn_full.py` — complete normalized reference kernel with explicit local-event handoff for the final `accum_ub` store/reuse edge
8. `agent/example/kernels/a2/flash_attn_full_pj_hif8.py` — normalized reference with `to_hif8_torch(p * 128) / 128`, validated non-aligned `S2` tail handling, and explicit local-event handoff for the final `accum_ub` store/reuse edge

### Generated host-side scalar dtype or project generation
1. `easyasc/kernelbase/kernelbase.py`
2. templates/resources under `easyasc/resources/`
3. related parser handlers if the generation path crosses lowering

### Test organization or fixture lookup
1. `historical testcases/README.md` (removed from this skill bundle)
2. relevant tests or fixtures under `historical automated tests/` (removed from this skill bundle)
3. `agent/example/demo/a2/` or `agent/example/demo/a5/` only if the target is intentionally manual or integration-style

## Reading Rule

Prefer:
1. one focused constraint/reference file
2. one implementation-path file or directory from this map
3. one test or source example

Do not jump into broad contributor architecture docs first unless the smaller path still leaves the question ambiguous.
