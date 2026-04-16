# Counter and Buffering Constraints

Read this file when a kernel uses `DBuff`, `TBuff`, nested loops, delayed reuse, or autosync-sensitive slot ownership.

## Goal

Keep buffer lifetimes explicit enough that:
- ownership stays understandable
- slot lineage stays stable
- autosync grouping stays correct
- different stages do not silently fight over one counter

## 1. Core rule

Different buffer lifetimes must use different counters.

This is not an optional cleanup preference.
It is a required authoring rule in this repository.

Symmetric rule:
- same-lifetime paired buffers may share one counter

## 2. What counts as the same lifetime

Buffers may share one counter when they are:
- loaded together
- consumed together
- retired together
- logically one stage pair

Typical example:
- `l1x` and `l1y` for the same `k` tile may share one `l1_cnt`

Buffers should not share a counter when one of them lives across a different loop boundary or stage boundary.

Typical example:
- `l1` streaming inside the `k` loop should not share the same counter with an outer `L0C` or vec postprocess stage

Concrete nested matmul example:
- if `l1x` / `l1y` advance on every `k` tile and are consumed as one operand pair, they may share one `l1_cnt`
- if `l0c` is owned by the outer `n` tile, keep its `l0c_cnt` separate from the `k`-loop `l1_cnt`
- reusing one counter across those different loop-owned lifetimes can blur autosync slot lineage and break the pipeline rhythm

## 3. Why this matters

Reusing one counter across different lifetimes can break reasoning in several ways:
- DBuff parity stops matching the real stage lifetime
- autosync grouping sees a misleading slot lineage
- nested loops become harder to reason about
- a later refactor silently changes which stage owns a slot

A kernel may still "look" fine while the lifetime model is already broken.

## 4. Preferred counter layout

Name counters by stage ownership, not by generic sequence order.
Good examples:
- `l1_cnt`
- `l0c_cnt`
- `tile_cnt`
- `stage1_cnt`
- `stage2_cnt`

Bad pattern:
- one global counter reused everywhere because it was convenient once

## 5. Nested-loop rule

Across different loop levels, do not reuse one counter for different `DBuff` / `TBuff` lifetimes.
If two loop levels each own their own buffering rhythm, give them separate counters.

This matters especially when:
- one loop streams operands
- a parent loop owns an output tile
- a delayed consumer runs one iteration later than the producer

## 6. Delayed-stage rule

If a stage consumes data one iteration later, treat that as a distinct lifetime unless the entire slot family truly remains one coherent delayed pipeline.

Stable pattern in this repository:
- one counter for immediate score/probability production
- a different counter for delayed `p @ value` accumulation

Do not hide a delayed lifetime behind the producer counter.

## 7. Same-pipe grouping still matters

Counter correctness is easier to preserve when same-pipe instructions are grouped together.
Within one loop iteration, prefer:
- grouped MTE loads
- grouped compute
- grouped writeback

This makes ownership and stage boundaries much more obvious.

## 8. Quick checklist

Before accepting the counter design, verify:
- each counter has one clear stage owner
- different loop-owned lifetimes do not share one counter
- same-lifetime pairs share only when that pairing is intentional
- delayed consumers use a counter that matches their own lifetime
- autosync grouping still matches the logical slot story

## Files to study

- `agent/example/kernels/a5/matmul_mknk_2dgrid_splitn.py`
- `agent/example/kernels/a5/matmul_mknk_2dgrid_splitk.py`
- `agent/example/kernels/a5/matmul_mknk_2dgrid_splitk_add1.py`
- `agent/example/kernels/a5/matmul_rowwise_norm_large_nk.py`
- `agent/example/kernels/a5/vec_cube_vec_scale2_abs_add1_matmul.py`
- `agent/example/kernels/a5/test_mla_entire.py`
