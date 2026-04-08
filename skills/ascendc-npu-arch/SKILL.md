---
name: ascendc-npu-arch
description: Ascend NPU 架构知识查询技能。通过芯片型号映射、架构代际划分和 arch35 特性说明，帮助判断目标平台能力、特性支持与条件编译策略。当需要确认芯片型号、NpuArch/SocVersion、架构差异、特性支持或编译分支条件时使用。
---

# Ascend NPU 架构知识

## 架构代际概述

| 概念 | 说明 |
|-----|------|
| **NpuArch** | 芯片架构，定义指令集和微架构 |
| **SocVersion** | 片上系统版本，软件命名标识 |

## 架构映射表

| NpuArch | __NPU_ARCH__ | 目录 | 芯片型号 |
|---------|---------------|------|---------|
| DAV_1001 | 1001 | arch32 | Ascend910 |
| DAV_2002 | 2002 | arch32 | Ascend310P |
| DAV_2201 | 2201 | arch32 | Ascend910B, Ascend910_93 |
| DAV_3002 | 3002 | arch32 | Ascend310B |
| DAV_3510 | 3510 | arch35 | Ascend950DT, Ascend950PR |

## Ascend950 (arch35) 独有特性

| 特性 | 说明 | 典型算子 |
|-----|------|---------|
| Regbase 编程 | 直接操作寄存器 | 量化算子 |
| SIMT 编程 | 线程级并行 | 随机数、排序 |
| FP8 格式 | 8-bit 浮点 | 量化、动态量化 |

## 详细文档

- [完整架构指南](references/npu-arch-guide.md)
