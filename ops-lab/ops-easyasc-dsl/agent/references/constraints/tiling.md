# Tiling Constraints

Read this file when a kernel needs non-trivial tile selection, core split selection, or local-buffer capacity reasoning.
Do not read it for tiny untiled examples.

## Goal

Choose a tile strategy that is:
- legal for local-buffer capacity
- compatible with downstream vec-side dependency
- stable for ownership and tail handling
- justified from the real repository rules

## 1. Separate two decisions

Treat these as separate layers:
1. tile shape: `TILE_M`, `TILE_N`, `TILE_K`
2. core split: `m_split`, `n_split`

Do not collapse them into one vague "best tiling" decision.

## 2. Default ownership pattern

For standard cube-major matmuls, split by tile index, not by raw row range:
- `tile_m = CeilDiv(M, TILE_M)`
- `tile_m_per_core = CeilDiv(tile_m, GetCubeNum())`
- iterate `mt in [tile_m_begin, tile_m_end)`

If the chosen strategy returns both `m_split > 1` and `n_split > 1`, map `GetCubeIdx()` onto a 2D split grid and partition both axes explicitly.

For batched independent small matmuls, flatten batch-like dimensions first and split directly over the flattened batch axis:
- `BHN = B * H * N`
- `bhn_per_core = CeilDiv(BHN, GetCubeNum())`
- iterate `bhn_idx in [bhn_begin, bhn_end)`
This is simpler than tile splitting when each item is one full small matrix pair.

## 3. Use the estimator when the choice is not obvious

Use `agent/scripts/estimate_matmul_datamove.py` when:
- the matmul is large enough that core split matters
- there are multiple legal tile candidates
- downstream vec work constrains which axis may be split

Read the estimator result as:
- tile shape
- core split
- loop mode
- candidate tie set, if any

Choose `split_mode` from downstream dependency, not only from cube-side datamove:
- `split_m`: keep all `N` tiles for the same `M` rows together on one core
- `split_n`: keep all `M` tiles for the same `N` columns together on one core
- `mix`: both axes may be split because vec-side work does not impose a one-axis ownership constraint

If two candidates tie on datamove, break the tie with downstream local-memory fit instead of taking the first row mechanically.
Concrete example: for `x @ y.t() + 1.0`, both `256x128x256` and `128x256x256` may tie on datamove for the large shape; the latter is the better fit because the vec stage can reuse a `64x256` UB tile directly.

## 4. Device-specific capacity and core count

Buffer budgets and core count differ between a2 and a5. Check the target device before choosing tile sizes.

| Resource | a2 (`b3`) | a5 (`950`) |
|----------|-----------|------------|
| Cube core count | **20** | **32** |
| L0A | 64 KB | 64 KB |
| L0B | 64 KB | 64 KB |
| **L0C** | **128 KB** | **256 KB** |
| **UB** | **192 KB** | **256 KB** |
| L1 | 512 KB | 512 KB |

Source: `easyasc/globvars.py` (defaults) and `easyasc/a5.py` (overrides). Core counts are in `easyasc/simulator/shared/runtime_helpers.py`.

Key consequences:
- a5 tile strategies that fit `DBuff L0C` (e.g. `TILE_M=128, TILE_N=256 → 128*256*4*2 = 256 KB`) will **overflow on a2** (`128 KB`).
- a5 core split with `GetCubeNum()=32` becomes `20` on a2; verify load balance for the smaller core count.
- a5 vec-side `DBuff UB` allocations up to `256 KB` must stay within `192 KB` on a2.

Practical rule: when writing an a2 kernel, always verify `L0C DBuff` first. `DBuff` allocates 2 slots, so a single `l0c` allocation needs `2 * TILE_M * TILE_N * sizeof(dtype) <= l0c_cap`.

## 5. Capacity rules are mandatory authoring checks

Before blaming the simulator, check the real buffer budgets.

Repository authoring rule (shared across devices):
- `L0A` budget: `64 KB`
- `L0B` budget: `64 KB`

Device-specific budgets (see section 4 above):
- `L0C` budget: `128 KB` (a2) / `256 KB` (a5)
- `UB` budget: `192 KB` (a2) / `256 KB` (a5)

Convert element counts into bytes using the real local-buffer dtype size.

### `splitk` capacity model
- `L0A` elements: `TILE_M * SPLIT_K * 2`
- `L0B` elements: `TILE_N * SPLIT_K * 2`
- the extra `* 2` is the DBUF factor

### `splitn` capacity model
- `L0A` elements: `TILE_M * TILE_K * 2`
- `L0B` elements: `SPLIT_N * TILE_K * 2`
- the extra `* 2` is the DBUF factor

### Byte-budget rule
- `l0a_bytes = l0a_elements * l0a_dtype_size`
- `l0b_bytes = l0b_elements * l0b_dtype_size`
- require both `l0a_bytes <= 64 * 1024` and `l0b_bytes <= 64 * 1024`

## 6. Choosing `splitk` vs `splitn`

Use the overflowing side to choose the split mode.
Do not switch blindly.

Choose `splitk` when:
- K-side staging into `L0A` / `L0B` is too large
- you want to keep a large outer `TILE_K` for strategy selection but legalize the real inner load size

Choose `splitn` when:
- N-side staging is too large
- output-tile width pushes the buffer budget too far

Hard rule:
- keep both `splitk` and `splitn` at `>= 32`
- do not push either one below `32`

Fallback rules:
- if `splitk` fails even at `32`, retile `TILE_M` / `TILE_N`
- if `splitn` fails even at `32`, try `splitk` instead of pushing `splitn` lower

## 7. Large-`K` validated pattern in this repository

Stable pattern for aligned `MKNK` half-input matmul:
- keep outer `TILE_K=256` for datamove/core-split selection
- legalize the real inner buffer budget with `SPLIT_K=64`

Concrete validated example:
- `TILE_M=128`
- `TILE_N=256`
- `TILE_K=256`
- `SPLIT_K=64`

Byte check:
- `L0A = 128 * 64 * 2 * 2 = 32 KB`
- `L0B = 256 * 64 * 2 * 2 = 64 KB`

Practical rule:
- choose `m_split` / `n_split` from the outer tile
- use `splitk` only to legalize the inner cube staging

## 8. `L0C` authoring rule

Treat non-zero `L0C` row offsets on matmul destinations as unsupported in authoring.
Even though the simulator has an offset path, the repository rule is:
- keep matmul destinations anchored at row offset `0`
- solve oversized `M` with a higher-level `TILE_M` decision instead of `l0c[row_offset:..., ...]`

`N`-side subdivision is still fine when the destination remains anchored.

## 9. Quick checklist

Before accepting a tiled kernel, verify:
- **target device identified** (a2/a5) and device-specific budgets used
- tile shape chosen explicitly
- core split chosen explicitly, using the correct core count for the device
- split mode matches downstream dependency
- `L0A` / `L0B` byte budgets checked
- **`L0C` DBuff total** checked against the device-specific `l0c_cap`
- **`UB` DBuff total** checked against the device-specific `ub_cap` (when vec stages exist)
- `splitk` or `splitn` kept at `>= 32`
- `L0C` destination remains row-offset `0`
- ownership and counters still make sense for the chosen loop structure

## Files to study

- `agent/scripts/estimate_matmul_datamove.py`
- `agent/scripts/tools_summary.md`
- `agent/example/kernels/a5/matmul_mknk_2dgrid_splitn.py`
- `agent/example/kernels/a5/matmul_mknk_2dgrid_splitk.py`
- `agent/example/kernels/a5/matmul_mknk_2dgrid_splitk_add1.py`
- `agent/example/kernels/a5/matmul_kmkn_blockwise_quant128.py`
