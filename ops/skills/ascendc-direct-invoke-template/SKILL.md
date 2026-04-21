---
name: ascendc-direct-invoke-template
description: Kernel直调工程模板，用于创建 Ascend C Kernel 直调工程项目。提供经过验证的样例工程和清晰的修改指南。触发：当用户需要创建 Kernel 直调工程、学习 Ascend C 编程、快速原型验证、或提及"Kernel直调"、"<<<>>>内核调用"时使用本 skill。
---

# Ascend C Kernel 直调工程

## 使用方法

1. **复制样例目录**：
   ```bash
   # 若算子目录<your_op>未创建
   cp -r references/add_custom <your_op>
   # 若算子目录<your_op>已存在
   cp -r references/add_custom/* <your_op>
   cd <your_op>
   ```

2. **全局替换算子名**：`add_custom` → `<your_op>`（add_custom 是整体算子名，_kernel/_torch/_tiling 是固定后缀）

3. **阅读代码中的注释**（搜索 `[MODIFY]` 标记），修改以下内容：
   - 类名和 kernel 函数名
   - Tiling 结构体（`add_custom_tiling.h`）
   - 计算逻辑（`add_custom_kernel.asc`）
   - 输入/输出数量
   - `CMakeLists.txt` 中的目标名

4. **编译运行**：
   ```bash
   # 完整流程（含编译）
   bash run.sh

   # 仅运行测试，复用已有编译产物（代码审查阶段使用，避免重复编译）
   bash run.sh --skip-build
   ```
   > `run.sh` 在运行 kernel 前会自动删除旧的 `output/output.bin`，确保精度验证读取的是本次运行的新鲜输出。

## 文件结构

模板按职责分目录，PyTorch 对接开箱即用：

```
├── op_kernel/               NPU 计算层
│   ├── add_custom_tiling.h      Tiling 常量 + 结构体（纯 C/C++，kernel 和 host 共用）
│   └── add_custom_kernel.asc    纯 kernel 代码（KernelAdd 类 + add_custom_kernel 核函数入口）
├── op_host/                 Host 直调层
│   ├── add_custom.asc           Host + main 入口（#include "add_custom_kernel.asc"）
│   └── data_utils.h             数据读写工具
├── op_extension/            PyTorch 接入层
│   ├── add_custom_torch.cpp     PyTorch host 实现（Tiling 计算 + kernel launch）
│   ├── register.cpp             TORCH_LIBRARY 注册（含 Meta backend）
│   └── ops.h                    函数声明
├── scripts/                 测试脚本
│   ├── gen_data.py               生成输入数据
│   ├── golden.py                 Golden 计算函数（直调 & PyTorch 双通路共用）
│   ├── verify_result.py          直调通路精度验证
│   └── test_torch.py             PyTorch 通路测试
├── CMakeLists.txt           双 target：可执行文件 + libadd_custom_ops.so
├── run.sh                   一键运行（支持 --torch 跑 PyTorch 通路）
└── README.md
```

## PyTorch 对接

模板已内置 PyTorch 对接，编译后即可使用：

```python
import torch
import torch_npu

torch.ops.load_library("build/libadd_custom_ops.so")
y = torch.ops.npu.add_custom(x1, x2)
```

## 代码关键模式

在 `add_custom_kernel.asc` 和 `add_custom.asc` 中可直接学习：

- **内存分配**: `TPipe` + `TQue` 管理 UB Buffer
- **数据流**: CopyIn → Compute → CopyOut 三段模式
- **同步**: `EnQue/DeQue` 确保操作顺序
- **Host 流程**: ACL 初始化 → KernelCall → 资源释放

## 参考资源

- [Ascend C 示例代码](https://gitcode.com/cann/asc-devkit/tree/master/examples)
- NPU 架构配置详见 `ascendc-npu-arch` skill
- PyTorch 对接详见 `torch-ascendc-op-extension` skill
