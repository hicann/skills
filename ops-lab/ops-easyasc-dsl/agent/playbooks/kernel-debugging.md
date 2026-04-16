# Kernel Debugging Playbook

Use this playbook when an existing kernel is wrong, unstable, warning-heavy, or unclear.
Debug in layers.
Do not jump between random fixes.

## Goal

Find the first broken assumption.
Fix the model, then fix the kernel.
Do not keep stacking patches on top of an unclear design.

## Debug order

Use this order unless there is a very obvious failure point:
1. contract and cast order
2. layout and shape bindings
3. tile and capacity assumptions
4. tail handling
5. sync and ownership
6. counters and lifetime separation
7. precision boundaries
8. parser/simulator/codegen implementation path

## 1. Re-check the exact contract

Verify the kernel against the real PyTorch formula.
Common failure modes:
- wrong cast order
- wrong transpose interpretation
- wrong reshape meaning
- accidental semantic drift from the intended formula

If the reference is still fuzzy, stop here and clarify it before changing the DSL code.

## 2. Re-check layout and shape binding assumptions

Verify:
- tensor logical shapes
- transpose site
- `shape_bindings`
- repeated scalar dimension mapping

Common signs:
- output shape is right but values are wrong everywhere
- only some shapes fail
- changing `M`, `N`, or `K` flips behavior unpredictably

Repository reminder:
- if repeated scalar dimensions are ambiguous, try explicit `shape_bindings` before deeper kernel surgery

## 3. Re-check tile and capacity assumptions

When the kernel is tiled, verify:
- `TILE_M`
- `TILE_N`
- `TILE_K`
- `m_split`
- `n_split`
- `splitk` or `splitn`
- `L0A` and `L0B` byte budgets
- `L0C` footprint

Repository reminders:
- keep `splitk` and `splitn` at `>= 32`
- choose `splitk` when K-side staging is too large
- choose `splitn` when N-side staging or output tile is too large
- do not author non-zero `L0C` row offsets on matmul destinations

If tile search is non-trivial, use `agent/scripts/estimate_matmul_datamove.py` instead of eyeballing it.

Read next if needed:
- `agent/references/constraints/tiling.md`

## 4. Re-check tail handling

Look at GM boundaries first, not local tensor sizes.
Common rule:
- local buffers stay full-tile sized
- only GM read/write boundaries use `valid_m`, `valid_n`, `valid_k`

For cube -> vec writeback, verify the standard half-row split:
- `half_rows = CeilDiv(valid_m, 2)`
- `row_begin = GetSubBlockIdx() * half_rows`
- `row_end = Min(row_begin + half_rows, valid_m)`

Symptoms of tail bugs:
- aligned cases pass but odd sizes fail
- only the last tile is wrong
- one vec subblock is correct and the other is garbage

Read next if needed:
- `agent/references/constraints/tail-safety.md`

If the kernel is a normalized online softmax with running `row_max` / `row_sum`,
also read:
- `agent/references/constraints/online-softmax-tail.md`

## 5. Re-check sync ownership

Assume ownership is wrong until proven otherwise.

Repository reminders:
- `auto_sync()` only manages same-side ordering
- `auto_sync()` does not replace cross-side ownership transfer
- cube -> vec handoff still needs `CvMutex`
- vec -> cube handoff still needs `VcMutex`

If the issue smells like pipeline ordering:
- inspect where the producer finishes
- inspect where the consumer starts
- check whether `lock/ready/wait/free` surround the real ownership edge
- keep the critical section narrow

Read next if needed:
- `agent/references/constraints/autosync.md`

## 6. Re-check counters and lifetimes

Many broken kernels are actually lifetime bugs.
Verify:
- which loop owns each buffer family
- whether different lifetimes accidentally share one counter
- whether the same slot lineage is expressed consistently

Repository reminders:
- buffers with different lifetimes must use different counters
- same-lifetime paired buffers may share one counter
- reusing one counter across different loop-owned lifetimes can silently break autosync grouping and slot reasoning

Read next if needed:
- `agent/references/constraints/counters.md`

## 7. Re-check precision boundaries

Verify where values change dtype.
Common failure modes:
- casting too early
- reducing in the wrong dtype
- writing packed or quantized data too early
- comparing against a reference with a different cast order

Repository reminder:
- keep matmul accumulation in `float`
- downcast later unless the design proves otherwise

Read next if needed:
- `agent/references/constraints/precision.md`

## 8. Inspect the real implementation path

If a rule is still unclear, inspect the actual implementation path instead of theorizing.
Before reading any device-specific branch, confirm the family mapping:
- `950` -> C310
- `b*` -> C220
- in `easyasc/resources/tensorutils.h`, `__DAV_C310__` is the `950` path
- in `easyasc/resources/tensorutils.h`, `__DAV_C220_CUBE__` is the `b*` path

Typical targets:
- `easyasc/stub_functions/`
- `easyasc/parser/`
- `easyasc/parser/asc_autosync.py`
- `easyasc/simulator/`
- `easyasc/shortcuts/matmul.py`

Good debugging question:
- which exact instruction gets emitted
- how the parser lowers it
- how the simulator executes it
- whether the kernel assumption matches that path

When the simulator itself produces an unexpected error or behavior:
- investigate the simulator path first before patching the kernel DSL
- inspect the exact simulator stage, runtime view, and lowered instruction that failed
- do not assume the upper-layer kernel is wrong just because the simulator failed first

If the simulator behavior still looks inconsistent with the intended repository model after real inspection:
- stop blind upper-layer edits
- summarize the concrete simulator finding
- pause and discuss the issue with the user before deciding whether to change the kernel, parser, or simulator

## 9. Build a minimal reproducer

When the full kernel is noisy, isolate one mechanism:
- one matmul
- one handoff
- one vec postprocess
- one autosync chain
- one tail tile

A minimal reproducer is usually faster than staring at a fused kernel.

## 10. Treat warnings as real signals

Do not accept a passing result with unresolved warnings.
Especially for `auto_sync`, warnings usually mean the lifetime model is off.

If a warning persists after real inspection, stop blind iteration.
Either redesign the stage boundary or ask the user for clarification.

## Symptom-to-check map

### Wrong everywhere
Check:
- formula
- transpose/layout
- cast order
- shape bindings

### Only large shapes fail
Check:
- tile budgets
- split mode
- estimator choice
- counter ownership across nested loops

### Only tail tiles fail
Check:
- `valid_*` handling
- half-row vec writeback split
- GM boundary slicing

### Autosync warnings or weird pipeline stalls
Check:
- same-side vs cross-side misunderstanding
- event family grouping assumptions
- counter reuse across different lifetimes
- unsupported instruction not covered by autosync pairing

### Simulator passes, generated path looks suspicious
Check:
- parser lowering path
- codegen handlers
- explicit event or mutex placement
- assumptions hidden by simulator convenience

### V2 simulator: CvMutex / VcMutex timeout (wait_vec / wait_cube)

When V2 reports a sync timeout such as:
```
wait_vec timeout: {'scope': 'intra_core', 'name': 'vec_to_cube_0_0',
                   'target_phase': 3, 'current_phase': 2, 'consumed_phase': 2}
```

The timeout almost always means the **other lane's actor thread crashed**, not that the sync logic itself is wrong.
The crashing thread silently terminates, so the expected `vec_ready` or `cube_ready` signal is never published, and the waiting side eventually times out.

Debugging sequence:
1. **Capture the real error on the other lane first.**
   Patch `CoreRuntime.start` to wrap each `ControlActor.start()` in a try/except that prints the lane name and exception.
   The first non-timeout error is the real root cause.
2. Common root causes behind the silent crash:
   - **Float8 indexing**: PyTorch does not support `tensor[indices]` for `float8_e5m2`/`float8_e4m3fn` dtypes. Any `_gather_1d`, `_scatter_1d`, or fancy indexing on a float8 register or UB tensor will raise `"index_cpu" not implemented for 'Float8_e5m2'`. Fix by viewing as `torch.uint8` before indexing.
   - **Non-contiguous UB views in burst copy**: `ub_to_gm_pad` uses `.view(torch.uint8)` on the source tensor. If the UB tensor is a column slice (e.g. `scalebuf[0:rows, 0:1]`) with stride > 1, `view()` fails. Fix by using `_linear_view_from_pointer()` to recover the flat physical byte layout.
   - **Micro op not implemented**: A `@vf()` body calls an op that V2's `MicroRuntime` does not dispatch, raising `NotImplementedError`. The vec lane dies and its `free()` never fires.
3. After fixing the vec/cube error, the sync timeout resolves on its own.
4. Do **not** tune sync timeouts or phase counters to work around these failures — the counters are correct; the lane just never ran to completion.

### V2 simulator: do not run multiple simulator processes in parallel

Running multiple V2 simulator processes concurrently on the same machine can produce **silent data corruption** — wrong results with no error or warning.

Root cause: the V2 simulator uses per-lane `PipeWorker` threads internally. Under heavy CPU thread contention (many processes × many threads per process), OS thread scheduling becomes aggressive enough to expose intra-process data races between pipe worker threads sharing the same tensor memory. The exact mechanism is timing-dependent and difficult to reproduce with a single process.

Affected kernels: primarily those using NZ layout operations (`ub_to_l1_nz`, `deinterleave`, `reg_to_ub`) or complex `@vf` functions. Simpler ND-only kernels are less likely to be affected but are not guaranteed safe.

The legacy simulator (`simulator="legacy"`) does not have this issue because it executes all pipe instructions sequentially on a single thread per lane.

Rules:
- **Always run kernel simulator tests sequentially**, not in parallel with `&` or batch scripts.
- If you must validate many kernels, run them one at a time in a `for` loop.
- If a kernel produces incorrect results only when run alongside other tests, re-run it alone before investigating the kernel itself.

### V2 simulator: known shape-dependent limitations

Some V2 issues only manifest with specific test shapes. Before deep-diving, try these workarounds:
- **Multi-K-tile accumulation**: When `K > TILE_K`, matmul results may be incorrect due to L1 buffer reuse across K tiles. Workaround: use `K == TILE_K` in test shapes.
- **Unaligned N causing CvMutex phase mismatch**: Non-TILE_N-aligned N values can cause the vec lane to execute fewer CvMutex cycles than the cube lane expects. Workaround: ensure N is divisible by TILE_N or BLOCK_N.
- **Ambiguous scalar binding**: When multiple kernel scalars have the same value (e.g. M == N == 256), `OpExec` auto-binding can pick the wrong tensor dimension. Fix: add explicit `shape_bindings` or use distinct dimension values.
- **Multi-head iteration race on UB**: When BH > 1, the vec lane's output write (ub_to_gm) from the previous head can race with init_buffers for the next head because `bar_all()` only drains each lane's own pipe queue, not cross-lane tasks. Fix: add `bar_all()` before `init_buffers` inside the bh loop so that each lane drains its pending pipe tasks (including any in-flight ub_to_gm) before zeroing accumulators. Also move counter declarations (`stage1_cnt`, `stage2_cnt`) outside the bh loop to avoid re-creating Var objects each iteration.

## Fallback references

Read these if more detail is needed:
- `agent/references/code-paths.md`
- `doc/11_architecture_for_contributors.md`
