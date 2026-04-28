# Transpose API 最佳实践

本文档聚焦 small-channel transpose 中常见的 API 组合、硬约束和反模式。

> **当前覆盖范围**：本文档当前仅覆盖**小通道 transpose**；大通道及通用 transpose 场景暂未包含，后续可按需要补充。

---

## 1. 核心计算链路

### 1.1 用 `TransDataTo5HD + Gather` 做小通道 transpose

```cpp
constexpr uint32_t EPB16 = 16;
uint32_t repeats = tileNA / 16;

LocalTensor<half> srcList[16];
LocalTensor<half> dstList[16];
for (uint32_t i = 0; i < 16; ++i) {
    srcList[i] = halfLocal[(i < channelCount) ? (i * tileNA) : 0];
    dstList[i] = vnLocal[EPB16 * i];
}

uint16_t dstRS = (repeats == 1) ? 0 : EPB16;
uint16_t srcRS = (repeats == 1) ? 0 : 1;
TransDataTo5HDParams params(false, false, static_cast<uint8_t>(repeats), dstRS, srcRS);
TransDataTo5HD<half>(dstList, srcList, params);
```

`TransDataTo5HD` 的输出每个 16-half block 只前 `channelCount` 个位置有效；剩余位置是 padding。后续必须再用 `Gather` 取出有效值。

### 1.2 `Gather` 提取有效通道

```cpp
auto halfOut = halfLocal;
Gather(halfOut, vnLocal, offsetLocal, 0, validCount);
Cast(outLocal, halfOut, RoundMode::CAST_ROUND, validCount);
```

如果前面已经在 FP32 阶段完成了 in-place round，那么这里的 `Gather` / `Cast` 往往会按对齐后的 count 处理，最终 `half -> uint8` 也可以直接使用 `CAST_NONE`；有效输出范围仍然由当前 tile 的 `curN * channelCount` 决定。

这里的 `offsetLocal` 是 host 端预计算好的 byte offset 表，对应：

```cpp
for (uint32_t p = 0; p < tileNA; ++p) {
    for (uint32_t c = 0; c < channelCount; ++c) {
        offset[p * channelCount + c] = (p * 16 + c) * sizeof(half);
    }
}
```

---

## 2. API 级硬约束

### 2.1 `Gather` 不直接处理 `uint8`

推荐路线是：

```text
FP32 -> half -> TransDataTo5HD -> Gather(half) -> uint8
```

不要尝试直接在 `uint8` 上做 gather 抽取。

### 2.2 `repeats == 1` 时 stride 必须置 0

```cpp
uint16_t dstRS = (repeats == 1) ? 0 : 16;
uint16_t srcRS = (repeats == 1) ? 0 : 1;
```

这是小 tile 场景的硬约束，不能省。

### 2.3 `VECOUT` depth 必须 >= 2

即便 `Compute` 逻辑看起来是“算完立刻写”，也不要把 `VECOUT` 队列缩成 1。多 tile 下 CopyOut 与后续 Compute 交错时，单槽位容易卡死流水。

### 2.4 非对齐写回优先 `DataCopyPad`

输出是 `curN * channelCount` 字节。只要不是严格 32 字节对齐，就优先：

```cpp
DataCopyPad(yGm[gmOffset], outLocal, copyParams);
```

不要为了少写一个 `Pad` 路径，引入额外的尾块分支复杂度。

---

## 3. 反例与反模式

| 反模式 | 问题 | 建议替代 |
|------|------|---------|
| `GetValue / SetValue` 逐元素搬运 | 标量 UB 读写，吞吐极差 | `DataCopy / DataCopyPad + vector route` |
| 逐像素 `DataCopyPad(blockLen=channelCount)` | DMA setup 成本远大于有效负载 | 按通道整段搬运，再做 `vnchwconv + Gather` |
| 默认套用通用 transpose API | 小通道场景下内部开销可能远大于实际计算 | 走专门的小通道路径 |
| 直接 `float -> half -> uint8` | 容易出现量化 off-by-1 | 先 in-place round，再转 half |
| 跨 tile 自己管理一次性 event | 容易把流水写成一次性同步死锁 | 用 `TQue` 的 `EnQue/DeQue` 管理 |
