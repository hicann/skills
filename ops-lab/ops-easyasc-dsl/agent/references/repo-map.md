# Repository Map

Use this file when the question is about where something lives in the repository.
Do not use it as a replacement for kernel-authoring or debugging guidance.

## Top-Level Areas

- `skill/`
  - repository skill entrypoint that calls into the workflow under `agent/`
- `easyasc/`
  - core DSL, parser/codegen, simulator, runtime models, and public API surface
  - restored on demand by `agent/scripts/init.sh`
- `agent/example/kernels/`
  - curated kernel examples plus legacy kernel summaries/tricks
  - restored on demand by `agent/scripts/init.sh`
- `historical automated tests/` (removed from this skill bundle)
  - automated pytest coverage plus reusable fixtures under `historical testcases/fixtures/` (removed from this skill bundle)
- `agent/scripts/`
  - small standalone utility scripts
- `doc/`
  - English project and API documentation
  - restored on demand by `agent/scripts/init.sh`
- `doc_cn/`
  - Chinese mirror of the documentation set
  - restored on demand by `agent/scripts/init.sh`
- `agent/`
  - easyasc DSL to AscendC workflow, including the router-first guidance layer, focused references, machine-readable catalogs, scripts, assets, and archived examples
- `agent/example/demo/`
  - manual runners and compile/integration demos grouped by device family
  - restored on demand by `agent/scripts/init.sh`
  - `agent/example/demo/a2/` holds a2-specific runnable samples
  - `agent/example/demo/a5/` holds a5/general runnable samples
  - negative-case repros currently live under `agent/example/demo/a5/negative_cases/`
- `README.md`, `README_CN.md`
  - top-level project and documentation entry points
- `SIMULATOR_V2_DESIGN.md`, `SIMULATOR_V2_TODO.md`, `SIMULATOR_V2_INSTRUCTION_TODO.md`
  - root-level planning docs for the V2 simulator architecture, phased rollout, and legacy instruction porting backlog
- `AGENTS.md`
  - repository-local working rules for agent-style contributors

## `agent/` Layout

- `agent/ROUTER.md`
  - first task router
- `agent/scripts/`
  - workflow utilities, repository maintenance, and initialization helpers
- `agent/assets/`
  - archived runtime/docs and archived examples restored by `agent/scripts/init.sh`
- `agent/playbooks/`
  - short workflow guides for common tasks
- `agent/references/constraints/`
  - detailed invariants and repository-specific rules
  - includes vec mask semantics in `agent/references/constraints/mask.md`
  - includes a2-specific device constraints in `agent/references/constraints/a2-device.md`
  - includes vec stride/slicing rules in `agent/references/constraints/vec-stride.md`
  - includes a2 vec reduction pattern in `agent/references/constraints/vec-reduction-a2.md`
- `agent/references/patterns/`
  - common pipeline topologies and shape patterns
  - includes a2-specific mixed-pipeline routes such as `a2-cube-vec.md`, `a2-cube-vec-cube.md`, `a2-cube-vec-cube-vec.md`, and `a2-cube-vec-cube-vec-softmax.md`
- `agent/references/examples/`
  - human-readable kernel/tool catalogs
- `agent/references/repo-map.md`
  - this file
- `agent/references/code-paths.md`
  - common implementation-path lookup guide
- `agent/index/`
  - generated machine-readable catalogs from the example markdown files

## Where To Look By Question

### I need to write a kernel
- start: `agent/playbooks/kernel-authoring.md`
- then: `agent/references/constraints/`
- then: `agent/references/patterns/`
- then: `agent/references/examples/kernel-catalog.md`
- only after that: `agent/example/kernels/` source files

### I need to debug a kernel
- start: `agent/playbooks/kernel-debugging.md`
- then: matching file under `agent/references/constraints/`
- then: `agent/references/code-paths.md`
- then: parser/simulator/source files

### I need a concrete example kernel or tool
- start: `agent/references/examples/kernel-catalog.md`
- start: `agent/references/examples/tool-catalog.md`
- machine-readable fallback: `agent/index/kernels.json`, `agent/index/tools.json`

### I need repo structure or ownership
- start: `agent/references/repo-map.md`
- deeper fallback: `doc/11_architecture_for_contributors.md`

### I need to know which code path implements an operation
- start: `agent/references/code-paths.md`
- deeper fallback: `doc/11_architecture_for_contributors.md`

### I need to understand test coverage layout or fixtures
- start: `historical testcases/README.md` (removed from this skill bundle)
- then: relevant tests under `historical automated tests/` (removed from this skill bundle)

### I need a manual runnable example or compile demo
- start: `agent/example/demo/`
- then pick the device bucket under `agent/example/demo/a2/` or `agent/example/demo/a5/`
- for intentionally failing repros: `agent/example/demo/a5/negative_cases/`

### I need to modify docs
- start: `agent/playbooks/doc-authoring.md`
- then: `README.md`, `README_CN.md`, `doc/`, `doc_cn/`

### I need to modify a tool
- start: `agent/playbooks/tool-authoring.md`
- then: `agent/references/examples/tool-catalog.md`
- then: `agent/scripts/`

## Owner Files

Use the smallest owner file that answers the question:

- contributor-level architecture and subsystem ownership -> `doc/11_architecture_for_contributors.md`
- implementation-path lookup -> `agent/references/code-paths.md`
- kernel examples and heuristics -> `agent/references/examples/kernel-catalog.md`
- tool summaries -> `agent/scripts/tools_summary.md`
- test-suite organization -> `historical testcases/README.md` (removed from this skill bundle)

## Legacy Files

These files still exist, but they are no longer the default first read:
- `agent/scripts/tools_summary.md`

Use them only when the smaller router-first layers are not enough.
