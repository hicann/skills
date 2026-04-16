# Agent Router

Use this file as the first routing layer for repository knowledge.
Do not treat it as a giant summary.
Its only job is to send the reader to the smallest useful next file.

Before following routes that need archived runtime, docs, or examples, run:
- `bash agent/scripts/init.sh`

## Primary Routes

### 1. Write a new kernel
Start with:
- `agent/playbooks/kernel-authoring.md`

Then drill down only if needed:
- tile and split decisions -> `agent/references/constraints/tiling.md`
- sync and event reasoning -> `agent/references/constraints/autosync.md`
- counter ownership -> `agent/references/constraints/counters.md`
- precision boundaries -> `agent/references/constraints/precision.md`
- datamove recipes -> `agent/references/constraints/datamove.md`
- vec reduction idioms -> `agent/references/constraints/reduction.md`
- normalized online softmax `S2` tails -> `agent/references/constraints/online-softmax-tail.md`
- pure a2 vec-only kernels -> `agent/references/constraints/a2-vec-kernel.md`
- topology-specific shape -> `agent/references/patterns/`

If the target device is a2 (b3), additionally read:
- `agent/references/constraints/a2-device.md`
- `agent/references/patterns/a2-cube-vec.md` (for cube -> vec topology)
- `agent/references/patterns/a2-cube-vec-cube.md` (for cube -> vec -> cube with delayed consumer)
- `agent/references/patterns/a2-cube-vec-cube-vec.md` (for cube -> vec -> cube -> vec with delayed numerator accumulation)
- `agent/references/patterns/a2-cube-vec-cube-vec-softmax.md` (for normalized online softmax with running sum and final divide)
- `agent/references/constraints/vec-reduction-a2.md` (for row-wise reductions)
- `agent/references/constraints/vec-stride.md` (for continuous vs sliced vec ops)

Fallback:
- `agent/references/code-paths.md`
- `doc/11_architecture_for_contributors.md`

### 2. Debug an existing kernel
Start with:
- `agent/playbooks/kernel-debugging.md`

If the issue is very specific, then drill down into the matching detailed reference.
Typical topics include:
- autosync -> `agent/references/constraints/autosync.md`
- tiling -> `agent/references/constraints/tiling.md`
- counters and buffering -> `agent/references/constraints/counters.md`
- precision and cast boundaries -> `agent/references/constraints/precision.md`
- tail handling -> `agent/references/constraints/tail-safety.md`
- normalized online softmax tails -> `agent/references/constraints/online-softmax-tail.md`
- datamove issues -> `agent/references/constraints/datamove.md`
- reduction bugs -> `agent/references/constraints/reduction.md`
- V2 simulator sync timeout or float8 issues -> see "V2 simulator" sections in `agent/playbooks/kernel-debugging.md`

Fallback:
- `agent/references/code-paths.md`
- `doc/11_architecture_for_contributors.md`

### 3. Find a reference example
Start with:
- kernel examples -> `agent/references/examples/kernel-catalog.md`
- tool examples -> `agent/references/examples/tool-catalog.md`

If machine-readable lookup helps, use:
- `agent/index/kernels.json`
- `agent/index/tools.json`

Fallback only when the new catalog still lacks the case you need:
- `agent/scripts/tools_summary.md`
- `agent/example/kernels/`
- `agent/scripts/`

### 4. Modify or add a repository tool
Start with:
- `agent/playbooks/tool-authoring.md`

Then drill down only if needed:
- tool selection / intended outputs -> `agent/references/examples/tool-catalog.md`
- machine-readable tool metadata -> `agent/index/tools.json`
- implementation summary / CLI examples -> `agent/scripts/tools_summary.md`

Fallback while the new playbook is still incomplete:
- `agent/scripts/tools_summary.md`
- `agent/references/code-paths.md`
- `doc/11_architecture_for_contributors.md`

### 5. Modify or extend repository documentation
Start with:
- `agent/playbooks/doc-authoring.md`

Fallback while the new playbook is still incomplete:
- `README.md`
- `README_CN.md`
- `doc/`
- `doc_cn/`
- `doc/11_architecture_for_contributors.md`

### 6. Understand repository structure
Start with:
- `agent/references/repo-map.md`

If you need deeper architecture detail after that:
- `doc/11_architecture_for_contributors.md`

### 7. Find the implementation path for a behavior or operation
Start with:
- `agent/references/code-paths.md`

If the smaller map is not enough:
- `doc/11_architecture_for_contributors.md`
- relevant source/test files

## Reference Layers

When a playbook is not enough, read only the specific detailed reference you need.
Detailed reference areas now available:
- `agent/references/constraints/tiling.md`
- `agent/references/constraints/autosync.md`
- `agent/references/constraints/counters.md`
- `agent/references/constraints/precision.md`
- `agent/references/constraints/tail-safety.md`
- `agent/references/constraints/online-softmax-tail.md`
- `agent/references/constraints/datamove.md`
- `agent/references/constraints/reduction.md`
- `agent/references/constraints/a2-device.md`
- `agent/references/constraints/a2-vec-kernel.md`
- `agent/references/constraints/vec-stride.md`
- `agent/references/constraints/vec-reduction-a2.md`
- `agent/references/patterns/`
- `agent/references/examples/`
- `agent/references/repo-map.md`
- `agent/references/code-paths.md`
- `agent/index/kernels.json`
- `agent/index/tools.json`

## Reading Rule

Prefer this order:
1. router
2. one playbook
3. one focused reference
4. one example catalog entry or source file

Do not load every legacy summary by default if a smaller route already answers the task.
