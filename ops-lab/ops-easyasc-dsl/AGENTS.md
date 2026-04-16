- Start every conversation by reading `skill/SKILL.md` first, then follow it into `agent/ROUTER.md`
- Follow the router-first reading order:
  1. read `skill/SKILL.md`
  2. read `agent/ROUTER.md`
  3. read only one matching playbook for the current task when possible
  4. read only the focused constraint, pattern, example, or map file that the playbook/router points to
  5. read source files only after the smaller guidance layers stop being enough
- Do not start by loading giant summaries when a smaller route already answers the task
- High-level owner files:
  - `agent/references/repo-map.md`: top-level repository layout and ownership map
  - `agent/references/code-paths.md`: implementation-path lookup for operations and behaviors
  - `doc/11_architecture_for_contributors.md`: contributor-level architecture and subsystem ownership fallback
  - `agent/scripts/tools_summary.md`: tool fallback; prefer `agent/references/examples/tool-catalog.md` and `agent/playbooks/tool-authoring.md` first
  - `historical testcases/README.md` (removed from this skill bundle): test-suite layout, fixture placement, and demo boundary
- For kernel authoring tasks:
  - start with `agent/playbooks/kernel-authoring.md`
  - then read only the specific constraint files you need
  - then use `agent/references/examples/kernel-catalog.md` to choose source examples
- For kernel debugging tasks:
  - start with `agent/playbooks/kernel-debugging.md`
  - then read only the matching constraint file for the suspected failure mode
- For tool changes:
  - start with `agent/playbooks/tool-authoring.md`
  - use `agent/references/examples/tool-catalog.md` before opening tool code
- For documentation changes:
  - start with `agent/playbooks/doc-authoring.md`
  - then use `README.md`, `README_CN.md`, `agent/references/repo-map.md`, and `doc/11_architecture_for_contributors.md` as needed
- For repo-structure questions:
  - start with `agent/references/repo-map.md`
  - then use `agent/references/code-paths.md` if the question is really about implementation location
  - then use `doc/11_architecture_for_contributors.md` only when the smaller maps are not enough

## Repository working rules

- Before guessing, inspect the real implementation path for the target behavior inside this repository.
  - check the relevant files under `easyasc/stub_functions/`
  - then parser lowering under `easyasc/parser/`
  - then simulator behavior under `easyasc/simulator/` when execution semantics matter
- Existing kernels are for study, not copy-paste. Use them to understand repository semantics, constraints, and synchronization patterns.
- Unless the user explicitly requests otherwise or the task is already tied to an existing kernel file, implement a new kernel in a new file by default.
- In this repository, "kernel" means a function decorated with `@kernel`. User-visible inputs and outputs must map strictly to the `GMTensor` arguments and returned `GMTensor` objects of that `@kernel` function.
- Build kernels incrementally: reason about tile strategy first, then implement and validate stage by stage instead of trying to finish the whole kernel in one jump.
- Justify each operation, cast, buffer, synchronization edge, and datamove from the target formula and hardware behavior. Do not keep a step only because another kernel happened to use it.
- Before golden inputs enter the kernel, only shape-only transforms are allowed, such as `squeeze`, `unsqueeze`, and `reshape`. Do not use `expand`, `tile`, `permute`, or other layout-changing transforms unless the user explicitly wants that behavior.
- If any part of the reasoning is ambiguous or cannot be justified from repository evidence, stop and ask instead of guessing.
- Treat warnings as signals that the mental model is incomplete. Investigate the root cause instead of accepting a passing result with unresolved warnings.
- For `auto_sync` warnings specifically, understand the synchronization model first, then either adjust the kernel or propose a concrete parser/autosync change.

## Update owner docs, not giant summaries

When stable repository knowledge changes, refresh the owner files that actually describe that area:
- repository structure or subsystem ownership -> `agent/references/repo-map.md` and/or `doc/11_architecture_for_contributors.md`
- implementation-path lookup -> `agent/references/code-paths.md`
- kernel additions or meaningful kernel changes -> `agent/references/examples/kernel-catalog.md`
- tool additions or tool behavior changes -> `agent/scripts/tools_summary.md`
- test organization, fixtures, or demo boundaries -> `historical testcases/README.md` (removed from this skill bundle)
- documentation entry maps -> `README.md`, `README_CN.md`, and the affected `doc/` or `doc_cn/` pages

- Prefer an existing local Python or conda environment that is already configured for Ascend/CANN work. `torch210npu` is only an example, not a required environment name.
- Typehints should be compatible with python3.8.
- All codes, error messages, and readme files should be written in English.
- Refer to `agent/example/kernels/` for templates.

## Python formatting

For Python code, do not automatically rewrite function signatures or calls into a one-argument-per-line style.

- Prefer compact formatting.
- Keep arguments on the same line if the result is still easy to read.
- Wrap only when necessary for readability or line length.
- If wrapping is needed, use a compact multi-line layout instead of placing every argument on its own line unless there is a clear readability benefit.
- Avoid unnecessary trailing commas that encourage vertical expansion.
