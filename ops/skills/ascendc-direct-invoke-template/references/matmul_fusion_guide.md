# MatMul 融合 Epilogue 深度指南

> 适用范围：`mxfp8 × mxfp8 matmul + eltwise`（Div/Mul/Add/Relu/Cast/组合算子）。
> 本文档用于统一 MatMul 融合分支在设计、开发、审查三阶段的规范。

## 0. 阅读路径与文档状态

### 0.1 角色阅读路径

- Architect：重点阅读“强约束规范（契约 A-H）”和“机制说明”。
- Developer：优先按“快速执行路径”落地，再按“契约 A-H”自检。
- Reviewer：以“契约 A-H + 提交前核查清单”为主。

### 0.2 文档状态定义

- `[PATTERN]`：固定结构，禁止改动。
- `[USER]`：按算子语义填写。
- `[SAMPLE]`：Div+float 样例取值，迁移时必须重评。

---

## 1. 快速执行路径（Developer）

### 1.1 最小改动步骤

1. 复制 `include/epilogue/div_epilogue.h` 到 `<your>_epilogue.h`。
2. 修改 `operator()` 中 Step c 的计算语句。
3. 在 `src/matmul_fused_swat.cpp` 替换 `#include` 与 `using MyEpilogue = ...`。
4. 在 `scripts/gen_data.py` 同步修改 `OUTPUT_DTYPE`、`gen_fusion_inputs()`、`compute_golden()`。
5. 编译运行并进行精度验证。

### 1.2 最小可运行检查

```bash
rg -c "void Init\(|auto GetTensor|void operator\(\)" include/epilogue/*_epilogue.h
rg "CrossCoreWaitFlag" include/epilogue/
rg -c "params\.(problemShape|mmadParams|l1Params|schParams|qbmmParams|epilogueParams)" src/*.cpp
rg "__cube__" src/matmul_fused_*.cpp
```

期望：
- 第一条输出为 3。
- 第二条、第四条为空。
- 第三条输出为 6。

---

## 2. 强约束规范（契约 A-H）

| 契约 | 约束内容 | 核验方式 |
| --- | --- | --- |
| A | Epilogue 必须提供 `Init / GetTensor / operator()`，且签名兼容 | `rg` 检查三接口 |
| B | `matmul_kernel_fused.h` 的调用时机和流程不由业务改写 | 与参考模板对比 |
| C | L0C→UB 走 Fixpipe SPLIT_M；Epilogue 不允许 `CrossCoreWaitFlag` | Block/Epilogue 关键字检查 |
| D | Host 必须完整填充 `problemShape/mmadParams/l1Params/schParams/qbmmParams/epilogueParams` | Host 参数填充检查 |
| E | Tiling 字段到 Host Params 映射完整 | 对照 `tiling_swat.h` 与 Host |
| F | 数据链路闭环：脚本 -> Host -> Kernel -> Epilogue | 输入文件名与读取链路核对 |
| G | 跨脚本与代码的 dtype 一致 | `*DTYPE` 与 `sizeof(...)` 对照 |
| H | MIX 模式入口禁止 `__cube__` | `rg "__cube__"` 检查 |

### 2.1 提交前核查清单

- [ ] 契约 A：Epilogue 三接口齐全
- [ ] 契约 B：Kernel 模板未发生结构性偏移
- [ ] 契约 C：CrossCore 责任边界清晰
- [ ] 契约 D：Host Params 六字段完整
- [ ] 契约 E：Tiling 映射无缺失
- [ ] 契约 F：输入输出链路闭环
- [ ] 契约 G：dtype 一致
- [ ] 契约 H：MIX 模式约束满足

---

## 3. 机制说明

### 3.1 角色边界

| 层 | 文件 | 职责 |
| --- | --- | --- |
| Kernel | `include/kernel/matmul_kernel_fused.h` | tile 循环、AIC/AIV 分发、CrossCore 协调 |
| BlockMmad | `include/block/block_mmad_swat.h` | AIC 侧 MMAD 与 Fixpipe（L0C→UB） |
| Epilogue | `include/epilogue/*_epilogue.h` | AIV 侧融合计算、写回 GM、发送 `AIV->AIC` |

Epilogue 不负责：
- `CrossCoreWaitFlag(AIC->AIV)`。
- `Fixpipe L0C->UB`。
- tile 主循环。

### 3.2 三接口合约（Kernel 依赖）

```cpp
class MyEpilogue {
public:
    struct Params {};
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    __aicore__ inline void Init(Params const&, int64_t, int64_t, ProblemShape&);
    __aicore__ inline auto GetTensor();
    __aicore__ inline void operator()(BlockShape const&, int64_t, int64_t);
};
```

### 3.3 UB 布局与 SPLIT_M

- `cLocal_`（matmul 输出）固定在 UB offset 0。
- `stageSize_` 必须满足对齐与分 stage 约束。
- SPLIT_M 场景必须追加：

```cpp
offset += AscendC::GetSubBlockIdx() * halfM * N;
```

### 3.4 CV 同步与 Hard Event

- Kernel 负责 AIC->AIV 等待与通知。
- Epilogue 仅在末尾发送 `CrossCoreSetFlag(AIV->AIC)`。
- AIV 内部阶段同步采用 `MTE3_MTE2 -> MTE2_V -> V_MTE3` 顺序。

### 3.5 DataCopyPad 与 stride

`curN < N` 时必须使用 `DataCopyPad + DataCopyExtParams` 显式设置 `srcGap/dstGap`。
`srcGap/dstGap` 单位为字节，不是元素数。

---

## 4. 常见问题与修复

| 现象 | 根因 | 修复动作 |
| --- | --- | --- |
| AIV 卡死 | Epilogue 内错误执行 `CrossCoreWaitFlag` | 删除 Epilogue 内 Wait，保留 Kernel 同步 |
| M 后半区异常 | 漏写 `SubBlockIdx * halfM * N` 偏移 | 按 3.3 补齐偏移 |
| 边缘 tile 错行 | 未使用 `DataCopyPad` 处理 stride | 按 3.5 改造 |
| 随机值/越界 | `stageSize_` 计算或 stage 循环错误 | 重算并分 stage |
| drain 卡死 | 未发送 `AIV->AIC` flag | 在 `operator()` 末尾补齐 |

---

## 附录 A：[SAMPLE] 重评清单

| 编号 | 样例项 | 风险 | 处理要求 |
| --- | --- | --- | --- |
| S1 | `UB_SIZE` | 跨芯片 UB 容量不同 | 按目标芯片重定义 |
| S2 | `using DataType = float` | dtype 迁移后链路不一致 | 同步修改 alias 与 `sizeof` |
| S3 | `stageNum = 2` | 输入路数变化导致布局错配 | 按 1/2/3 路重设 |
| S4 | `ALIGN_ELEM = 32/sizeof(float)` | dtype 改变后对齐错误 | 改为 `sizeof(DataType)` |
| S5 | `rowBytes = ... * sizeof(float)` | stride 字节数错误 | 按输出 dtype 重算 |
| S6 | `divisor*` 命名 | 语义漂移 | 按业务重命名 |
| S7 | batch=1 假设 | 批量场景不成立 | 超范围场景单独扩展 |
| S8 | RowMajor 假设 | 布局变更导致偏移错误 | 保持 RowMajor 或全链路调整 |
| S9 | `sizeD = m*n*sizeof(...)` | 广播输入 shape 不匹配 | 同步改 Host 与 Epilogue |
| S10 | `input_d.bin` 文件名 | 语义不匹配 | 按算子语义改名并同步脚本 |

---

## 附录 B：派生算子改动矩阵

| 片段 | 归属 | 改动策略 |
| --- | --- | --- |
| `Init/GetTensor/operator` 框架 | `[PATTERN]` | 保持结构 |
| SPLIT_M 切分与 stage 循环 | `[PATTERN]` | 保持框架 |
| Step a GM->UB 输入读取 | `[USER]` | 按是否有第二输入保留/删除 |
| Step c 计算语句 | `[USER]` | 必改 |
| 末尾 `CrossCoreSetFlag` | `[PATTERN]` | 保持 |
| `stageNum` / `DataType` / `sizeof(float)` | `[SAMPLE]` | 必须重评 |

常见场景：
- Mul/Add：替换 Step c；保留双输入读取。
- Relu：删除第二输入读取；`stageNum` 通常为 1。
- Cast 输出：新增 cast 缓冲并同步 Host 与脚本 dtype。
- Div+Relu：两步计算并保留必要 `PipeBarrier<PIPE_V>`。

---

## 5. 维护约束（跨文档）

- `SKILL.md` 仅保留路由与入口，深度机制不重复描述。
- 设计模板与 task-prompts 引用本指南“章节标题”，不绑定易漂移编号。
- `div_epilogue.h` 保留最短替换提示，完整解释统一在本指南维护。
