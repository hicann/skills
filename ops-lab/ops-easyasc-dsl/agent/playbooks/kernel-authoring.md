# Kernel Authoring Playbook

Use this playbook when writing a new kernel or when replacing a kernel body in a major way.
Keep the workflow short, incremental, and explicit.

## Goal

Produce a kernel that:
- matches the exact PyTorch contract
- uses a justified topology
- respects repository-specific tiling, sync, precision, and tail rules
- is validated incrementally instead of guessed into existence

## 0. File layout and formatting

Kernel file layout order:
1. imports and file-level constants
2. helper `@vf` / `@func` blocks
3. main `@kernel` definition (upper half of file)
4. `__main__` with full validation story (sample inputs, inline PyTorch reference, `OpExec` launch, comparison)

Authoring rule for local helpers:
- if a helper is called from inside `@kernel`, decorate it explicitly with `@func()` (or `@vf()` where supported)
- leave plain Python helpers undecorated only when they are reference / test utilities that are never called from the kernel body

Do not bury the kernel body under long runner wrappers.
A reader should be able to see the reference formula and the `OpExec` launch without jumping elsewhere.

Kernel-site call formatting:
- for `@kernel`, keep input `GMTensor` args on one line, output on the next, `Var` args on the next when wrapping
- for `@vf` / `@func`, keep all args on one line when readable; if wrapping, `Tensor` args on one line and scalars on the next
- at `OpExec(...)` call sites, keep tensor inputs on one line, scalar dims on one line, `shape_bindings=` on its own line when needed
- keep `matmul(...)`, `*_vf(...)`, `dst[...] <<= src[...]` on one line whenever still readable

## 1. Identify the target device before anything else

Determine whether the kernel targets a2 (`easyasc.a2`, device `b3`) or a5 (`easyasc.a5`, device `950`).
This affects buffer capacities and core count, which change tile selection and core split downstream.

Key differences:
- core count: a2 = `20`, a5 = `32`
- L0C: a2 = `128 KB`, a5 = `256 KB`
- UB: a2 = `192 KB`, a5 = `256 KB`
- L0A, L0B, L1: same (`64, 64, 512 KB`)

A tile strategy valid on a5 may overflow L0C or UB on a2.
Always check device-specific budgets before finalizing tile sizes.

a2-specific constraints beyond capacity:
- no `l0c_to_ub`, no `@vf`, no `Reg`/`RegList`
- cube → vec must go through GM workspace (`l0c_to_gm_nz2nd` → `gm_to_ub_pad`)
- `CvMutex` uses `dst_end_pipe=Pipe.MTE2` (not `Pipe.V`)
- each sub-block has independent 192 KB UB; use `GetSubBlockIdx()` to split work
- vec reduction (`cmax`) outputs dense scalars; must `brcb` before use in `sub`/`div`

Read next if needed:
- `agent/references/constraints/tiling.md` (section 4)
- `agent/references/constraints/a2-device.md` (a2-specific differences)
- `agent/references/constraints/a2-vec-kernel.md` (pure vec-only elementwise / bit-level kernels on a2)
- `agent/references/patterns/a2-cube-vec.md` (GM workspace bridge pattern)
- `agent/references/patterns/a2-cube-vec-cube.md` (double GM bridge with delayed consumer)
- `agent/references/patterns/a2-cube-vec-cube-vec.md` (triple bridge with delayed numerator accumulation)
- `agent/references/patterns/a2-cube-vec-cube-vec-softmax.md` (triple bridge with running sum and final divide)
- `agent/references/constraints/online-softmax-tail.md` (non-aligned `S2` rule for normalized online softmax)
- `agent/references/constraints/vec-stride.md` (continuous vs sliced vec ops)
- `agent/references/constraints/vec-reduction-a2.md` (cmax + brcb row-max pattern)

## 2. Start from the exact contract

Write the golden PyTorch formula first.
Keep cast order exact.
If the reference uses `einsum`, rewrite the index meaning into one or more matmul-style stages before touching the DSL.

Common einsum-to-matmul rewrites:
- `einsum('bhd,bshd->bhs')`: flatten `query` to `[BH,D]`, flatten `key` to `[BSH,D]` with `reshape` only, gather per-head key rows inside the kernel with strided `gm_to_ub_pad` before the `1xD @ SxD` matmul
- `softmax(einsum(...), dim=-1)`: keep the same reshape-only layout, then add explicit row-max and row-sum passes over the `S` tiles before the final cast
- `einsum('bhs,bshd->bshd')`: keep probabilities as `[BH,S]`, flatten `value` to `[BSH,D]`, and either gather scalar weights with strided `gm_to_ub_pad` or store `p.half()` into an internal GM workspace

If that softmax is a normalized online softmax over `S2` tiles:
- do not assume GM `valid_n` slicing is enough for a non-aligned tail
- invalid score columns must behave like `-inf` before `rowmax`
- read `agent/references/constraints/online-softmax-tail.md`

If scalar args repeat dimensions across tensors, set `shape_bindings` explicitly.
If you hit a shape ambiguity error, add `shape_bindings` and rerun once before changing the kernel body.

Do not start from an existing kernel body.
Study examples, but derive the new kernel from the target formula.

## 3. Choose the pipeline topology before coding

Classify the kernel first:
- cube only
- cube -> vec
- vec -> cube
- vec -> cube -> vec
- cube -> vec -> cube
- cube -> vec -> cube -> vec

This decision controls:
- where buffers live
- where ownership changes happen
- whether `CvMutex` or `VcMutex` is needed
- how many counters and stages are required

If the formula contains multiple matmuls, add them one stage at a time.
Do not fuse everything in the first attempt.

## 4. Fix layout and shape decisions early

Before implementing, decide:
- input tensor logical layout
- output logical layout
- whether the kernel is `x @ y.t()` or `x.t() @ y`
- whether repeated scalar dimensions require `shape_bindings`

Repository rules to remember:
- `x @ y.t()` usually means `x:[M,K]`, `y:[N,K]`
- `KM @ KN -> MN` usually means load transpose views only at the matmul call site
- if shape inference is ambiguous, add `shape_bindings` before changing the kernel body

## 5. Choose tile strategy before filling in the kernel body

For larger matmuls, determine:
- `TILE_M`
- `TILE_N`
- `TILE_K`
- `m_split`
- `n_split`
- whether `splitk` or `splitn` is required

Use `agent/scripts/estimate_matmul_datamove.py` when the tile/core split is non-trivial.
Choose split mode from downstream dependency, not only from cube-side datamove.

Repository rules to remember:
- separate tile shape from core split
- `splitk` and `splitn` must stay at or above `32`
- check `L0A` and `L0B` byte budgets explicitly before blaming the simulator
- do not author matmul destinations with non-zero `L0C` row offsets

Read next if needed:
- `agent/references/constraints/tiling.md`

## 6. Plan precision boundaries explicitly

Decide where each dtype change happens:
- source dtype
- matmul accumulation dtype
- vec or micro postprocess dtype
- final output dtype

Repository rule:
- keep matmul accumulation in `float` unless there is a very strong reason not to
- downcast later, usually in vec or micro stages

Read next if needed:
- `agent/references/constraints/precision.md`

## 7. Plan ownership, sync, and counters before implementation

For each stage, identify:
- producer side
- consumer side
- handoff buffer
- counter owner
- buffer lifetime

Repository rules to remember:
- `auto_sync()` is same-side ordering only
- cross-side ownership still needs explicit mutexes
- different lifetimes must use different counters
- same-lifetime paired buffers may share one counter
- keep same-pipe instructions grouped when possible

Typical cross-side mappings in this repository:
- cube -> vec: `CvMutex(..., src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)`
- vec -> cube: `VcMutex(..., src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)`

For accumulation semantics on writeback, wrap the GM store with `with atomic_add():`.
This works independently on cube and vec output paths.
Keep atomic sections narrow (around store only).

Read next if needed:
- `agent/references/constraints/autosync.md`
- `agent/references/constraints/counters.md`
- `agent/references/constraints/datamove.md`
- `agent/references/constraints/reduction.md`

## 8. Implement the smallest valid slice first

Preferred order:
1. build the shortest correct version of the formula
2. validate one stage in the simulator
3. add the next stage
4. validate again
5. only then add more fusion or optimization

If the final kernel is large, still begin from a smaller but structurally honest version.

If the target formula contains two or more matmuls, implement progressively:
1. start from one matmul and validate that baseline
2. add the remaining matmuls one by one, evaluating after each addition
3. keep previously validated stages unchanged while introducing the next matmul

## 9. Validate incrementally

Validation order:
1. check formula equivalence
2. run at least one aligned case
3. run at least one tail case
4. verify precision-sensitive outputs with the right tolerance
5. only then move on to more complex shapes

Keep the runnable story visible in `__main__`:
- sample input construction
- inline PyTorch reference
- `OpExec(..., simulator=True)` launch
- comparison output

Typical tolerance profile:
- float / half normalized paths: `rtol`/`atol` around `1e-3` to `3e-3`
- float8 quantized paths: looser checks on fp8 outputs (for example around `2e-1`)
- for KM/KN kernels and blockwise outputs, include alignment and divisibility guards (`%16`, `%128`)

## 10. Inspect the real implementation path when confused

If behavior is unclear, inspect the repository implementation instead of guessing.
Typical places:
- `easyasc/stub_functions/`
- `easyasc/parser/`
- `easyasc/parser/asc_autosync.py`
- `easyasc/simulator/`
- `easyasc/shortcuts/matmul.py`

Do not treat passing output with unresolved warnings as good enough.
Warnings usually mean the model is incomplete.

If simulator execution fails unexpectedly:
- inspect the simulator path before changing the kernel structure
- verify the lowered instruction, runtime tensor views, and simulator-side assumptions first
- do not rush into upper-layer kernel edits until the simulator failure mode is understood

## 11. Use examples intentionally

Study existing kernels for legal patterns, not for copy-paste.
Pick the example from the catalog first, then open the source file.

Start with:
- `agent/references/examples/kernel-catalog.md`

Good example targets by need:
- shortest cube baseline: `agent/example/kernels/a5/matmul_float_mmad.py`
- cube -> vec postprocess: `agent/example/kernels/a5/basic_cube_vec_mix.py`
- vec -> cube preprocess: `agent/example/kernels/a5/vec_cube_abs_sqrt_matmul.py`
- split-`k` large-`K` tiled matmul: `agent/example/kernels/a5/matmul_mknk_2dgrid_splitk.py`
- staged mixed pipeline: `agent/example/kernels/a5/vec_cube_vec_scale2_abs_add1_matmul.py`
- one-tile lookahead pipeline: `agent/example/kernels/a5/test_mla_entire.py`
- a2 cube-only baseline: `agent/example/kernels/a2/qk_matmul_batched.py`
- a2 cube -> vec with GM bridge and row-max: `agent/example/kernels/a2/flash_attn_score.py`
- a2 running state across tiles (dup init + vmax on [M,1]): `agent/example/kernels/a2/flash_attn_score_iter.py`
- a2 cube -> vec -> cube with one-tile lookahead and shared `L0C`: `agent/example/kernels/a2/flash_attn_score_pv.py`
- a2 cube -> vec -> cube -> vec with shared `L0C` and delayed numerator accumulation: `agent/example/kernels/a2/flash_attn_unnorm.py`
- a2 cube -> vec -> cube -> vec normalized online softmax with running sum/final divide: `agent/example/kernels/a2/flash_attn_full.py`

## 12. Stop and ask instead of guessing

Stop and ask the user when:
- the formula is ambiguous
- layout expectations are underspecified
- repeated shapes make multiple interpretations plausible
- a warning persists after real inspection
- simulator behavior appears inconsistent with the intended model after real inspection
- a correct implementation path cannot be justified from the repository evidence

## Fallback references

Read these if more detail is needed:
- `agent/references/code-paths.md`
- `doc/11_architecture_for_contributors.md`
