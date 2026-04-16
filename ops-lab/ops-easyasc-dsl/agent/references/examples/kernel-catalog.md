# Kernel Catalog

Use this file to choose the right kernel to study before opening source files.
This is a selection index, not a replacement for the actual kernel code.
Do not copy a kernel body directly just because the formula looks similar.

For each entry:
- `formula`: the main reference contract
- `topology`: the staged pipeline shape
- `study_for`: what this file is actually good for
- `do_not_copy_when`: when the resemblance is misleading

## Index schema

This file is also the machine-readable metadata source for `agent/index/kernels.json`.
The index builder reads:
- each `###` entry heading as one kernel record
- the surrounding `##` section as the kernel category
- top-level entry bullets such as `formula`, `topology`, `study_for`, and `do_not_copy_when`
- nested bullet items under those fields as ordered list values

If you edit this catalog, keep that field structure stable.

## Vec-only baselines

### `agent/example/kernels/a2/to_hif8_torch.py`
- formula: `y = to_hif8_torch(x)` with float32 output that emulates hif8 rounding/saturation exactly
- topology: `vec-only`
- study_for:
  - a2 pure-vec elementwise quantization without cube stages
  - exponent-bit extraction through `reinterpret` + `vand`/`vnot`
  - explicit `RoundMode.TRUNC` based implementation of `sign(x) * floor(abs(x) + 0.5)`
  - preserving `NaN`/`Inf` while handling underflow and overflow in-band
- do_not_copy_when:
  - you need true `DT.hif8` runtime output loading rather than float32 emulation
  - your kernel is fundamentally cube-bound or mixed cube/vec rather than vec-only

## Cube-only baselines

### `agent/example/kernels/a5/matmul_float_mmad.py`
- formula: `z = x @ y.t()`
- topology: `cube-only`
- study_for:
  - shortest end-to-end cube matmul baseline
  - minimal simulator validation story
  - first sanity check for pure cube lowering
- do_not_copy_when:
  - you need tiled DBuff structure
  - you need mixed cube/vec ownership
  - you need large-shape split selection

### `agent/example/kernels/a5/matmul_e5m2_shortcut.py`
- formula: `z = x.float() @ y.float().t()` with float8 inputs
- topology: `cube-only`
- study_for:
  - float8 input staging into float accumulation
  - minimal float8 guard pattern in the runnable section
- do_not_copy_when:
  - your problem is mainly about tiling, not dtype
  - you need vec-side postprocess or quantized output

### `agent/example/kernels/a5/matmul_kmkn_fp32_out.py`
- formula: `z = x.float().t() @ y.float()` with `x:[K,M]`, `y:[K,N]`
- topology: `cube-only`
- study_for:
  - transpose-at-matmul-call-site pattern
  - `KM @ KN -> MN` layout reasoning
  - explicit `%16` guards and `shape_bindings`
- do_not_copy_when:
  - your data is naturally `MKNK`
  - your main issue is mixed pipeline staging

### `agent/example/kernels/a5/matmul_mknk_2dgrid_splitn.py`
- formula: `z = x.float() @ y.float().t()` with `x:[M,K]`, `y:[N,K]`
- topology: `cube-only`
- study_for:
  - 2D core split for aligned `MKNK`
  - outer tile/core split selection with `splitn`
  - sharing one `l1_cnt` for one operand-pair lifetime
- do_not_copy_when:
  - K-side staging is the real capacity bottleneck
  - you need vec postprocess after cube output

### `agent/example/kernels/a5/matmul_mknk_2dgrid_splitk.py`
- formula: `z = x.float() @ y.float().t()` with `splitk`
- topology: `cube-only`
- study_for:
  - large-`K` aligned `MKNK` strategy
  - keeping outer `TILE_K=256` while legalizing inner staging with `SPLIT_K=64`
  - 2D split selection plus split-`k` accumulation ownership
- do_not_copy_when:
  - N-side width is the real issue and `splitn` is cleaner
  - your kernel is mainly about postprocess logic rather than cube staging

## Cube -> vec postprocess

### `agent/example/kernels/a5/basic_cube_vec_mix.py`
- formula: `z = abs(x @ y.t()) + 1.0`
- topology: `cube -> vec`
- study_for:
  - smallest mixed pipeline baseline
  - basic `CvMutex` ownership story
  - standard half-row vec writeback
- do_not_copy_when:
  - your kernel needs advanced tile scheduling
  - your vec stage has rowwise reductions or multiple outputs

### `agent/example/kernels/a5/matmul_half_splitn_bias10p2_vf.py`
- formula: `z = ((x.float() @ y.float()) + 10.2).half()`
- topology: `cube -> vec`
- study_for:
  - float accumulation followed by vec-side downcast
  - stable FIX->V handoff pattern
  - half-output writeback after vec postprocess
- do_not_copy_when:
  - your output should stay float
  - you need large-shape 2D split logic

### `agent/example/kernels/a5/matmul_rowwise_norm.py`
- formula: `z = (x @ y.t()) / row_sum(x @ y.t())`
- topology: `cube -> vec`
- study_for:
  - rowwise vec reduction after cube output
  - `cadd()` + `dup()` normalization pattern
- do_not_copy_when:
  - you need a two-pass large-`N` strategy
  - you need quantized or fp8 output

### `agent/example/kernels/a5/matmul_rowwise_norm_large_nk.py`
- formula: same as `matmul_rowwise_norm.py`
- topology: `cube -> vec`
- study_for:
  - two-pass normalization for larger `N/K`
  - temporary output persistence plus later reload
  - row-sum lifetime that spans multiple `N` tiles
- do_not_copy_when:
  - your normalized stage fits comfortably in one pass
  - your main problem is cube-side capacity rather than vec-side persistence

### `agent/example/kernels/a5/matmul_rowwise_l2_norm.py`
- formula:
  - `z = x.float() @ w.float().t()`
  - `out = z / sqrt(sum(z^2, dim=1, keepdim=True))`
- topology: `cube -> vec`
- study_for:
  - two-pass rowwise L2 normalization after matmul
  - per-row squared-sum accumulation across `N` tiles
  - explicit `SHAPE_BINDINGS` and aligned-shape guard in the Python wrapper
- do_not_copy_when:
  - your normalization is sum-based rather than L2-based
  - your shape is not naturally aligned to the validated contract (`M%64`, `N%256`, `K%128`)

### `agent/example/kernels/a5/matmul_chunk_absmax_norm128.py`
- formula: normalize each 128-column chunk by per-row absmax
- topology: `cube -> vec`
- study_for:
  - blockwise row statistics per fixed `CHUNK_N`
  - `abs -> cmax -> dup -> divide` idiom
- do_not_copy_when:
  - your block size is not naturally tied to cube `N` tiles
  - you need scale output rather than normalized values only

### `agent/example/kernels/a5/matmul_kmkn_blockwise_quant128.py`
- formula:
  - `z_tmp = x.float().t() @ y.float()`
  - `scale = absmax(z_tmp, block=128) / 224`
  - `z = (z_tmp / scale).to(e5m2)`
- topology: `cube -> vec`
- study_for:
  - blockwise scale generation
  - fp8 quantized output after float accumulation
  - optional `pack4()` fallback path
- do_not_copy_when:
  - you do not need quantized output
  - your layout is `MKNK` rather than `KMKN`

### `agent/example/kernels/a5/matmul_mknk_2dgrid_splitk_add1.py`
- formula: `z = x.float() @ y.float().t() + 1.0`
- topology: `cube -> vec`
- study_for:
  - large-`K` split-`k` cube stage with vec postprocess
  - estimator-driven 2D split selection plus vec tie-break concerns
  - separate counter for longer postprocess lifetime
- do_not_copy_when:
  - you only need a pure cube baseline
  - your vec stage is more complex than in-place elementwise add

### `agent/example/kernels/a5/cube_vec_atomic_add_two_outputs.py`
- formula:
  - `out_cube += x @ y.t()`
  - `out_vec += abs(x @ y.t()) + 1`
- topology: `cube -> vec` with dual outputs and atomics
- study_for:
  - atomics on both cube and vec writeback paths
  - mixed dual-output ownership
- do_not_copy_when:
  - you do not need atomic accumulation
  - you only have one output path

## Vec -> cube preprocess

### `agent/example/kernels/a5/vec_cube_abs_sqrt_matmul.py`
- formula: `z = x.float().abs().sqrt().half().float() @ y.float().t()`
- topology: `vec -> cube`
- study_for:
  - ND vec preprocess then cube consume
  - subblock row publish into `L1`
  - `VcMutex` ownership edge
- do_not_copy_when:
  - your preprocess should stay packed NZ end-to-end
  - your host-side contract already gives cube-ready input

### `agent/example/kernels/a5/vec_cube_abs_sqrt_matmul_nz.py`
- formula: same as `vec_cube_abs_sqrt_matmul.py`
- topology: `vec -> cube`
- study_for:
  - NZ publish path after vec preprocess
  - deinterleave + `reg_to_ub` packing before `l1 <<= ub.nz()`
- do_not_copy_when:
  - ND publish is enough and simpler
  - you do not actually need packed-NZ staging

### `agent/example/kernels/a5/recompute_wu_cube_vec.py`
- formula:
  - `k_cumdecay = attn.float() @ (k_beta * decay_exp).float()`
  - `kv = attn.float() @ v.float()`
- topology: `vec -> cube`
- study_for:
  - strict `[*,1]` scalar-broadcast preprocess via `single()`
  - dual cube outputs after one vec preprocessing stage
  - flattened batch-axis scheduling with `BHN`
- do_not_copy_when:
  - your dimensions are not close to this specialized recurrent/WU structure
  - you need a generic attention template rather than this specific dual-output path

## Vec -> cube -> vec fusion

### `agent/example/kernels/a5/vec_cube_vec_scale2_abs_add1_matmul.py`
- formula: `z = abs((x * 2).half().float() @ y.float().t()) + 1.0`
- topology: `vec -> cube -> vec`
- study_for:
  - one fused preprocess + cube + postprocess chain
  - explicit use of both `VcMutex` and `CvMutex`
  - stage separation with independent counters
- do_not_copy_when:
  - you have not yet validated the simpler vec->cube or cube->vec stage independently
  - your fusion requires delayed reuse across iterations

## Cube -> vec -> cube -> vec lookahead pipeline

### `agent/example/kernels/a5/test_mla_entire.py`
- formula: streamed MLA-style score, softmax, delayed `p @ k_nope`, and final normalization
- topology: `cube -> vec -> cube -> vec`
- study_for:
  - one-tile lookahead scheduling with warmup and drain
  - delayed-consumer counters (`stage1_cnt` vs `stage2_cnt`)
  - on-chip delayed reuse instead of forced GM round-trip
  - streamed `row_max` / `row_sum` / numerator accumulation
- do_not_copy_when:
  - your kernel does not truly need delayed stage reuse
  - you have not yet stabilized the simpler two-stage or three-stage version of the formula

### `agent/example/kernels/a5/mha_ifa.py`
- formula: streamed single-row attention `softmax(q @ k.t()) @ v`
- topology: `cube -> vec -> cube -> vec`
- study_for:
  - row-specialized `L=1` decode-style attention on a5
  - flattened `BH` scheduling with one query row kept resident while streaming `S`
  - simpler standard-attention lookahead flow than `agent/example/kernels/a5/test_mla_entire.py`
- do_not_copy_when:
  - you need multi-row query tiles
  - you need rope/nope fusion, fp8 staging, or MLA-specific math
  - your delayed stage cannot stay on chip cleanly

### `agent/example/kernels/a5/mha_ifa_256.py`
- formula: streamed single-row attention `softmax(q @ k.t()) @ v` with `BASES=256`
- topology: `cube -> vec -> cube -> vec`
- study_for:
  - keeping a `256`-wide on-chip score/value tile for half-input single-row attention on a5
  - using `splitk=64` for the `q @ k.t()` stage and `splitn=64` for the `p @ v` stage without shrinking the outer `BASES`
  - simpler ND baseline for `BASES=256` before trying NZ-published probability tiles
- do_not_copy_when:
  - your tile does not actually need a `256`-wide outer `S` chunk
  - you need multi-row query tiles
  - you have not first validated the simpler `BASES=128` path

### `agent/example/kernels/a5/mha_ifa_nz.py`
- formula: streamed single-row attention `softmax(q @ k.t()) @ v` with NZ-published probability tiles
- topology: `cube -> vec -> cube -> vec`
- study_for:
  - publishing vec-produced `p` tiles to `L1` in NZ layout for the delayed cube consumer
  - row-specialized `L=1` decode-style attention when stage 2 wants packed-NZ input
  - explicit `reg_to_ub(...).nz()` bridge inside a lookahead attention pipeline
- do_not_copy_when:
  - delayed stage 2 is fine with the simpler ND `l1p` path from `agent/example/kernels/a5/mha_ifa.py`
  - you need multi-row query tiles
  - your consumer does not actually benefit from packed-NZ staging

### `agent/example/kernels/a5/mha_ifa_nz_256.py`
- formula: streamed single-row attention `softmax(q @ k.t()) @ v` with `BASES=256` and NZ-published probability tiles
- topology: `cube -> vec -> cube -> vec`
- study_for:
  - widening the NZ-published `p` tile to `256` on a5 while keeping the lookahead decode-style schedule
  - splitting a `256`-wide half row into two `128`-lane micro registers before `ub_to_l1_nz`
  - pairing `splitk=64` / `splitn=64` with an NZ `l1p` handoff instead of the simpler ND path
- do_not_copy_when:
  - delayed stage 2 is fine with the simpler ND `l1p` path from `agent/example/kernels/a5/mha_ifa_256.py`
  - you need tail-safe `S` handling without a full `BASES`-wide GM slice
  - your consumer does not actually benefit from packed-NZ staging

## Pure vec and micro references

### `agent/example/kernels/a5/recurrent_state_attn_vec.py`
- formula: recurrent attention-state update specialized for `D=128`
- topology: `vec-only`
- study_for:
  - pure vec stateful update pattern
  - `RegList`-heavy row math
  - flattening `(B,H,S,D)` into vec-friendly layouts
- do_not_copy_when:
  - your kernel needs cube compute
  - your dimension pattern is not this specialized state update

### `agent/example/kernels/a5/vec_unaligned_gm_to_ub_pad.py`
- formula: vec compute on padded unaligned GM width (`exp + 2`)
- topology: `vec-only`
- study_for:
  - unaligned-width `gm_to_ub_pad` behavior
  - UB second-dim padding strategy
  - quick padded-transfer sanity checks
- do_not_copy_when:
  - your real problem is cross-side staging rather than vec padding

### `agent/example/kernels/a5/micro_cast_fp8_pack4_dual.py`
- formula:
  - `out_e5m2 = src.to(float8_e5m2)`
  - `out_e4m3 = src.to(float8_e4m3fn)`
- topology: `micro-only`
- study_for:
  - micro cast path
  - `RegLayout.ZERO` plus required `pack4()` squeeze before UB writeback
  - dual-fp8-output micro flow
- do_not_copy_when:
  - your kernel is mainly a cube or vec pipeline
  - you only need a single conventional cast without micro-specific layout concerns

## a2 kernels

### `agent/example/kernels/a2/sort_rows.py`
- formula: row-wise ascending sort that returns both sorted values and source indices
- topology: `vec-only`
- study_for:
  - a2 vec-only `sort32 -> mergesort4 -> mergesort_2seq` sorting pipeline
  - deinterleaving sorted value/index pairs with `gather`
  - multi-row vec-core partitioning with `GetVecNum()` and `GetVecIdx()`
  - validating sort output through both values and reconstructed gather-by-index roundtrip
- do_not_copy_when:
  - cube compute is involved
  - you only need a simple reduction instead of a full row sort
  - the row width is not aligned to the validated `4096`

### `agent/example/kernels/a2/qk_matmul_batched.py`
- formula: `qk = q.float() @ k.float().t()` with batched BH flattening
- topology: `cube-only`
- study_for:
  - simplest a2 kernel baseline
  - batched M-tile distribution with BH flattening
  - L0C capacity verification for a2 (128 KB)
- do_not_copy_when:
  - you need vec postprocessing
  - you target a5

### `agent/example/kernels/a2/flash_attn_score.py`
- formula: per-block `exp(Q @ K^T / sqrt(D) - row_max)` cast to half
- topology: `cube -> vec` (GM workspace bridge)
- study_for:
  - a2 cube → vec via GM workspace (no `l0c_to_ub`)
  - `CvMutex(FIX → MTE2)` cross-side synchronization
  - `split_workspace` with pingpong double-buffer `[CubeNum, 2, M, N]`
  - sub-block split with `GetSubBlockIdx()` for independent UB
  - `vmax → cmax → brcb → sub` row-max pattern on a2 vec
  - continuous vs sliced vec operation distinction
  - float → half output cast
- do_not_copy_when:
  - target is a5 (use `l0c_to_ub` + `@vf` instead)
  - no vec postprocessing needed
  - the reduction pattern differs from per-row max

### `agent/example/kernels/a2/flash_attn_score_iter.py`
- formula: per-block `exp(Q @ K^T / sqrt(D) - running_row_max)` with cross-tile max accumulation, cast to half
- topology: `cube -> vec` (GM workspace bridge)
- study_for:
  - running state accumulation across inner-loop iterations on a2
  - `dup(float('-inf'))` initialization for identity-element pattern (avoids conditional logic)
  - `vmax` on `[M, 1]` scalar format: why it covers all rows while `[M, 8]` does not
  - `dup` placement inside `auto_sync` outer loop (safe, generates extra V→MTE3 event)
  - incremental extension of an existing kernel (diff from `flash_attn_score.py` is 3 lines)
- do_not_copy_when:
  - you need full softmax (this is the unnormalized intermediate — no sum/divide pass)
  - you need per-tile independent max (use `flash_attn_score.py` instead)
  - target is a5 (use register-level running state instead)

### `agent/example/kernels/a2/flash_attn_score_pv.py`
- formula:
  - `score_j = q.float() @ k_j.float().t() * scale`
  - `m = maximum(m, rowmax(score_j))`
  - `p_j = exp(score_j - m).half()`
  - `pv_j = p_j.float() @ v_j.float()`
- topology: `cube -> vec -> cube` (double GM workspace bridge, one-tile lookahead)
- study_for:
  - a2 delayed-consumer pipeline with `n_loops + 1` warmup/drain schedule
  - reuse of one `L0C` family across two cube stages with one shared `l0c_cnt`
  - a2 `vec -> cube` bridge via `UB -> GM workspace -> L1` when `ub_to_l1_*` is unavailable
  - two-`workspace` design: float score bridge plus half probability bridge
  - preserving per-block running-max semantics while feeding the delayed `p @ v` cube stage
  - flattened output layout `[ (bh * n_tiles + tile_n) * S1 + row, D ]`
- do_not_copy_when:
  - you need normalized online softmax with running sum/divide
  - your target is a5 and direct `UB -> L1` publish is available
  - the second stage does not truly consume the vec result one iteration later
  - your `D` is not fixed/aligned to the validated `128`

### `agent/example/kernels/a2/flash_attn_unnorm.py`
- formula:
  - `score_j = q.float() @ k_j.float().t() * scale`
  - `curr_m = maximum(prev_m, rowmax(score_j))`
  - `expdiff_j = exp(prev_m - curr_m)`
  - `p_j = exp(score_j - curr_m).half()`
  - `pv_j = p_j.float() @ v_j.float()`
  - `out = out * expdiff_j + pv_j`
- topology: `cube -> vec -> cube -> vec` (triple GM bridge, one-tile lookahead)
- study_for:
  - a2 streamed unnormalized attention numerator with delayed final vec accumulation
  - reusing one physical `L0C` family across the two cube stages on a2
  - triple ownership edge: `CvMutex -> VcMutex -> CvMutex`
  - keeping running max, delayed `expdiff`, and final `accum` resident in vec UB
  - using one extra GM workspace for delayed `pv_j` because a2 cannot keep the stage-2 output on chip for vec reuse
  - safe copy pattern for `[M,1]` scalar state on a2 (`add(..., zero)` instead of `ub_to_ub`)
- do_not_copy_when:
  - you need normalized online softmax with running sum/final divide
  - your target is a5 and direct on-chip handoff is available
  - your second-stage output does not need to return to vec for delayed accumulation
  - your `D` is not fixed/aligned to the validated `128`

### `agent/example/kernels/a2/flash_attn_full.py`
- formula:
  - `score_j = q.float() @ k_j.float().t() * scale`
  - `curr_m = maximum(prev_m, rowmax(score_j))`
  - `expdiff_j = exp(prev_m - curr_m)`
  - `p_j = exp(score_j - curr_m)`
  - `row_sum = row_sum * expdiff_j + p_j.sum(-1)`
  - `pv_j = p_j.half().float() @ v_j.float()`
  - `out = out * expdiff_j + pv_j`
  - `out = out / row_sum`
- topology: `cube -> vec -> cube -> vec` (triple GM bridge, one-tile lookahead, final vec divide)
- study_for:
  - a2 normalized online flash attention with running `row_max` and running `row_sum`
  - preserving the exact `p_j.half().float()` value-path contract while keeping `row_sum` in float
  - reducing `sum_j` from the float probability tile before the cast
  - final sliced `div` of `[M,128]` accumulators by a narrow `[M,8]` row-sum broadcast
  - reusing the `flash_attn_unnorm.py` delayed numerator pipeline and extending it with full normalization
- do_not_copy_when:
  - you only need the unnormalized numerator (use `flash_attn_unnorm.py`)
  - your target is a5 and direct on-chip handoff is available
  - your contract does not require the exact `p.half().float()` value path
  - your `D` is not fixed/aligned to the validated `128`

### `agent/example/kernels/a2/flash_attn_full_pj_hif8.py`
- formula:
  - `score_j = q.float() @ k_j.float().t() * scale`
  - `curr_m = maximum(prev_m, rowmax(score_j))`
  - `expdiff_j = exp(prev_m - curr_m)`
  - `p_j = exp(score_j - curr_m)`
  - `row_sum = row_sum * expdiff_j + p_j.sum(-1)`
  - `p_q = to_hif8_torch(p_j * 128.0) / 128.0`
  - `pv_j = p_q.half().float() @ v_j.float()`
  - `out = out * expdiff_j + pv_j`
  - `out = out / row_sum`
- topology: `cube -> vec -> cube -> vec` (same triple bridge, scaled hif8 simulation in the stage-1 vec path)
- study_for:
  - preserving float `row_sum` while swapping the value path from `p.half().float()` to `to_hif8_torch(p * 128) / 128`
  - a2 non-negative hif8 simulation built from `compare_scalar(...) + select(...)` without relying on `uint8 -> float` casts
  - copying `[M,64]` slices into a contiguous scratch tile before `reinterpret(...)`, then writing the quantized chunk back
  - fitting a scaled hif8 probability path into the a2 softmax pipeline without changing the delayed stage-2 cube consume contract
  - extending normalized online softmax from aligned shapes to non-aligned `S2` by combining GM `valid_n` slicing with score-domain `-inf` masking before `rowmax`
  - using a vec-mask suffix invalidation pass on `[M,64]` score halves instead of trying to fix the tail only in the delayed `p` path
  - building the suffix-invalid bit pattern through a signed `int64` left-shift sequence, then assigning it into a `uint64` mask `Var` to avoid simulator-side unsigned-overflow issues
  - extending the same kernel to non-aligned `S1` by combining `valid_m` GM-boundary handling with a fixed-physical-subblock `local_valid_m` rule instead of an a5-style compact half split
  - handling `S1` tail rows differently from `S2` tail columns: invalid `S1` rows are zero-filled at `q` load, then overwritten with `-inf` only after `score - rowmax` and before `exp`, so delayed `p @ v` sees zero contribution while GM only receives valid rows
- do_not_copy_when:
  - your contract still wants the unscaled `p.half().float()` path (use `flash_attn_full.py`)
  - you need a generic float-domain hif8 kernel instead of the non-negative probability specialization
  - your `D` is not fixed/aligned to the validated `128`

## Fast selection hints

Use this quick map when you only need one starting point:
- a2 cube-only baseline -> `agent/example/kernels/a2/qk_matmul_batched.py`
- a2 cube -> vec with GM bridge -> `agent/example/kernels/a2/flash_attn_score.py`
- a2 running state across tiles -> `agent/example/kernels/a2/flash_attn_score_iter.py`
- a2 cube -> vec -> cube lookahead -> `agent/example/kernels/a2/flash_attn_score_pv.py`
- a2 cube -> vec -> cube -> vec delayed numerator accumulation -> `agent/example/kernels/a2/flash_attn_unnorm.py`
- a2 cube -> vec -> cube -> vec normalized online softmax -> `agent/example/kernels/a2/flash_attn_full.py`
- a2 normalized online softmax with scaled hif8 `p_j` path -> `agent/example/kernels/a2/flash_attn_full_pj_hif8.py`
- shortest cube baseline -> `agent/example/kernels/a5/matmul_float_mmad.py`
- transpose-layout baseline -> `agent/example/kernels/a5/matmul_kmkn_fp32_out.py`
- large-`K` split-`k` cube baseline -> `agent/example/kernels/a5/matmul_mknk_2dgrid_splitk.py`
- large-`K` split-`k` with vec postprocess -> `agent/example/kernels/a5/matmul_mknk_2dgrid_splitk_add1.py`
- simplest cube -> vec baseline -> `agent/example/kernels/a5/basic_cube_vec_mix.py`
- float -> half vec postprocess -> `agent/example/kernels/a5/matmul_half_splitn_bias10p2_vf.py`
- rowwise normalize -> `agent/example/kernels/a5/matmul_rowwise_norm.py`
- rowwise L2 normalize -> `agent/example/kernels/a5/matmul_rowwise_l2_norm.py`
- blockwise quantization -> `agent/example/kernels/a5/matmul_kmkn_blockwise_quant128.py`
- vec preprocess before cube -> `agent/example/kernels/a5/vec_cube_abs_sqrt_matmul.py`
- recurrent WU dual-output preprocess -> `agent/example/kernels/a5/recompute_wu_cube_vec.py`
- fused vec -> cube -> vec -> `agent/example/kernels/a5/vec_cube_vec_scale2_abs_add1_matmul.py`
- delayed lookahead mixed pipeline -> `agent/example/kernels/a5/test_mla_entire.py`
