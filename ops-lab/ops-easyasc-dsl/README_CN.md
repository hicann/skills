# ops-easyasc-dsl

[English README](README.md)

***有天我一拍脑袋想看看 AI 究竟能做成啥样，所以就有了这个仓。代码基本是纯添加零天然。我只是个傻读代码和指手画脚的人类罢了。***

`ops-easyasc-dsl` 提供 `easyasc` Python DSL，并把仓库整理成面向 skill 的结构，用统一的编写方式描述混合 Ascend 风格的 kernel。它可以：

- 从 Python 代码生成指令 IR
- 把 IR 下沉成拆分后的 cube/vec 代码路径
- 在内置模拟器中运行 kernel
- 为非模拟器场景生成 custom-op 源码产物

这个仓库主要围绕三件事展开：

1. 用 `@kernel`、`Tensor`、`GMTensor`、`Var` 以及可选的 `@vf` 微函数辅助，在 Python 里编写 kernel。
2. 由框架负责构建指令 IR、补上各执行侧需要的同步，并把程序拆成 cube 和 vec 两条路径。
3. 再通过模拟器或生成出来的运行时产物验证结果。

## 为什么会有这个仓库

这个代码库主要服务于 kernel 的开发、实验和调试，尤其适合下面这些场景：

- 快速验证新的纯 cube 或混合 cube/vec 流水线原型
- 在模拟环境里验证尾块处理、tiling 和精度边界
- 查已有 kernel，看看哪些 DSL 写法和实现套路是可行的

## 首次初始化

现在仓库把运行时代码和长文档打包放在 `agent/assets/ops-easyasc-dsl-runtime.tar.gz` 里，并把可运行示例打包放在 `agent/assets/ops-easyasc-dsl-example.tar.gz` 里。
新 checkout 后，如果要读取 `doc/`、`doc_cn/`、直接 import `easyasc.*`，或者运行 `agent/example/` 下的内容，先执行一次：

```bash
bash agent/scripts/init.sh
```

这个脚本是幂等的，会把归档里的 `easyasc/`、`doc/`、`doc_cn/` 和 `agent/example/` 还原出来。

## 快速开始

如果你本地已经有适用于 Ascend/CANN 的 conda 环境，优先激活它。`torch210npu` 只是一个示例：

```bash
conda activate torch210npu
```

然后运行一个最小可执行的 kernel 示例：

```bash
python agent/example/kernels/a5/matmul_float_mmad.py
```

这个示例现在会把两种执行模式都跑一遍：

1. 用 `@kernel` 定义一个 kernel
2. 通过 `OpExec(..., simulator=True)` 启动它
3. 再通过 `OpExec(..., simulator=False)` 启动它
4. 把两种输出都和 PyTorch 参考结果做对比

## `OpExec` 真机构建 + CANNSIM 所需环境变量

使用默认的 `simulator=False` 跑 `OpExec(kernel)` 时，框架会在**当前工作目录**生成并执行 `b.sh`、`r.sh`。脚本里已写入通用默认值；若本机布局不同，请在启动 Python 前自行导出下表变量（路径均为相对概念，不设具体机器目录）。

| 变量 | 何时需要设置 | 生成脚本中的典型默认 |
|------|----------------|----------------------|
| `ASCEND_HOME_PATH` | 当前 shell 里还拿不到 CANN 工具 | `<你的-cann-安装根>`（其下应有 `bin/setenv.bash`） |
| `ASCEND_CUSTOM_OPP_PATH` | 需要拼接额外自定义 OPP 根 | 可为空；导出后避免 `set -u` 与 `set_env.bash` 冲突 |
| `EASYASC_PYTHON_BIN` | CANN `opbuild` 调用的 `python3` 需要 **NumPy** | 含有该解释器的目录（如某 conda 环境的 `bin`），会加入 `PATH` |
| `PYTHONPATH` | 需从仓库根 import `easyasc` | 仓库根目录 |

`b.sh` / `r.sh` 用 **`EASYASC_ROOT`**（脚本所在目录）拼接相对路径；请在**仓库根目录**跑代码生成，使脚本落在根目录。只有在 `ASCEND_HOME_PATH` 已设置且有效时，脚本才会 `source "${ASCEND_HOME_PATH}/bin/setenv.bash"`，然后配置厂商库 `LD_LIBRARY_PATH`，并用 `cannsim` 跑 aclnn 测试二进制。

若 `op_build` 缺 `.so`，要么导出正确的 `ASCEND_HOME_PATH`，要么先把当前 shell 的 CANN 环境准备好；若报缺少 `numpy`，设置 `EASYASC_PYTHON_BIN` 指向带 NumPy 的 Python 所在目录。

## 核心概念

- `easyasc.a5`：面向 A5 风格架构与指令序列的一套公开 DSL 接口。当前仓库里大多数 kernel 和测试都在使用它，包含 cube、vec、micro、寄存器、cast 和调试辅助功能。
- `easyasc.a2`：面向 A2 风格架构与指令序列的另一套公开 DSL 接口。它不是 `a5` 的兼容层，而是针对另一类指令族的并列架构接口。
- `GMTensor`：全局内存张量，对应 kernel 的输入和输出。
- `Tensor`：位于 `L1`、`L0A`、`L0B`、`L0C` 或 `UB` 中的片上张量。
- `DBuff` / `TBuff`：用来描述基于 slot 复用的缓冲张量辅助类型。
- `Var`：用于表示循环边界、维度和符号形状的标量值。
- `OpExec`：运行时入口，可用于模拟器执行或代码生成。

## 典型开发流程

1. 先把精确的 PyTorch 公式写清楚。
2. 选择流水线拓扑：
   - 仅 cube
   - cube -> vec
   - vec -> cube
   - vec -> cube -> vec
   - cube -> vec -> cube
3. 用 Python 把 kernel 实现出来。
4. 使用 `OpExec(..., simulator=True)` 验证。
5. 如果重复出现的标量维度让形状推导变得不明确，就显式加上 `shape_bindings`。
6. 只有在模拟器结果和参考实现对齐之后，再继续看生成产物和硬件相关执行。

## 仓库结构

- `skill/`：仓库级 skill 入口，用来调用 `agent/` 下的 workflow
- `agent/`：easyasc DSL 到 AscendC 算子生成的 workflow，本体包含路由、脚本、参考资料和打包资源
- `agent/example/kernels/`：精选的示例 kernel，以及围绕 kernel 的说明文档；通过 `agent/scripts/init.sh` 按需恢复
- `historical automated tests/` (removed from this skill bundle)：模拟器、parser 和 codegen 回归测试
- `agent/example/demo/`：按设备家族整理的手动运行示例；通过 `agent/scripts/init.sh` 按需恢复
- `agent/example/demo/a2/`：a2 相关手动示例
- `agent/example/demo/a5/`：a5/通用手动示例，以及负例复现
- `agent/example/demo/a5/negative_cases/`：预期应当尽早失败的手动负例
- `easyasc/`、`doc/`、`doc_cn/`：通过 `agent/scripts/init.sh` 按需恢复
- `agent/references/examples/kernel-catalog.md`：逐个 kernel 的选择索引与学习指引

## 文档索引

如果当前工作区里还没有 `doc/`，先执行 `bash agent/scripts/init.sh`。

- [快速开始](doc/01_quickstart.md)
- [编程模型](doc/02_programming_model.md)
- [编写第一个 Kernel](doc/03_write_your_first_kernel.md)
- [混合流水线与同步](doc/04_mixed_pipeline_and_sync.md)
- [模拟器与 Trace](doc/05_simulator_and_trace.md)
- [代码生成与运行时](doc/06_codegen_and_runtime.md)
- [Kernel 模式与范式](doc/07_kernel_patterns.md)
- [测试与验证](doc/08_testing_and_validation.md)
- [API 参考](doc/09_api_reference.md)
- [故障排查](doc/10_troubleshooting.md)
- [面向贡献者的架构说明](doc/11_architecture_for_contributors.md)
- [Stub 与 Codegen 名字对照表](doc_cn/12_stub_to_codegen_name_map.md)
