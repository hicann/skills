// ============================================================
// [USER] Epilogue 骨架 — 融合算子 AIV 侧后处理（最小可填充版）
//
// 本文件是**三接口 Epilogue 骨架模板**。Developer 从这里起步，
// 把所有 `[USER]` 和 `[SAMPLE]` 标注位按算子语义填/替换。
//
// 与主样板 `div_epilogue.h` 的区别：
//   - 骨架用中性命名（`secondInputGlobal_` / `resultLocal_`），不带算子语义
//   - Step b 计算行默认保留 `AscendC::Div(...)` 作为示意，Developer 必须替换
//   - 关键常量 UB_SIZE / stageNum 显式引用 hardware_constants 或 constexpr，方便追踪
//
// 建议选择：
//   - 融合形式与 Div 相近（matmul 结果 + 一路第二输入 → eltwise → 输出）
//     → 直接复制 `div_epilogue.h` 作为主样板
//   - 无第二路输入（Relu/GeLU/Cast）或多路第二输入（Mul+Add）
//     → 从本骨架起步，按注释增删 UB 分区和 Step
//
// ------------------------------------------------------------
// 三接口合约（Kernel 层 `matmul_kernel_fused.h` 严格依赖）— 禁止改签名
// ------------------------------------------------------------
//   1) Init(params, baseM, baseN, problemShape4)
//   2) GetTensor() -> LocalTensor<DataType>
//   3) operator()(blockShape, gmOffset, flagId)
//
// 标注分层：
//   [PATTERN]  骨架必备 — 任何算子保留
//   [USER]     业务变量 — 必须填/替换
//   [SAMPLE]   Div/float 样例恰好的值 — 其他场景必须重新评估
//
// 关键约束（违反将导致 UB 越界 / 数据竞争 / 死锁）:
//   - UB offset 0 必须是 matmul 结果；不得占用
//   - 分 stage 循环处理（大 tile 单 stage 会 UB 越界）
//   - GM R/W 必须 `DataCopyPad + DataCopyExtParams`（curN < N 时有 srcGap）
//   - MTE3_MTE2 / MTE2_V / V_MTE3 三对 hard event 显式 Set/Wait
//   - 末尾必须 `CrossCoreSetFlag<MODE, PIPE_MTE3>(flagId)` 通知 AIC
//   - CV Wait 由 Kernel 层做，**不要**在 Epilogue 内部 Wait
// ============================================================
#ifndef EPILOGUE_SKELETON_H
#define EPILOGUE_SKELETON_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "epilogue/cv_sync_constants.h"
#include "utils/hardware_constants.h"
#include "kernel_utils/common_utils.h"

// [USER] 按算子命名（MulEpilogue / AddEpilogue / ReluEpilogue / ...）
class EpilogueSkeleton {
public:
    static constexpr uint16_t ZERO_FLAG = 0;
    static constexpr uint16_t AIC_SYNC_AIV_MODE_4 = CvSync::MODE;

    // [PATTERN] UB 物理大小来自 hardware_constants.h
    static constexpr uint32_t UB_SIZE = ::Hardware::UB_SIZE;

    // [SAMPLE] 当前骨架假定全链路 float（同 DivEpilogue）。
    //          若输出或第二路输入为 half / bf16 / int8 / bool，替换此 alias。
    using DataType = float;

    using BlockShape   = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

    // ---- [USER] Params: 声明本 Epilogue 需要的 GM 地址 ----
    // 典型模式：
    //   - 1 路第二输入（Div/Mul/Add/Bias）：`secondInputGmAddr + outputGmAddr`
    //   - 0 路第二输入（Relu/GeLU/Cast）：仅 `outputGmAddr`
    //   - 2 路第二输入（Mul+Add）：`inputXGmAddr + inputYGmAddr + outputGmAddr`
    struct Params {
        GM_ADDR secondInputGmAddr{nullptr};  // [USER] 重命名或删除
        GM_ADDR outputGmAddr{nullptr};
    };

    // ---- [PATTERN] UB Tensors ----
    // cLocal_       : matmul 结果（AIC Fixpipe 写入，UB offset 0）
    // secondLocal_  : [USER] 第二路输入缓冲（无则删）
    // resultLocal_  : 计算结果缓冲
    AscendC::LocalTensor<DataType> cLocal_{AscendC::TPosition::VECIN, 0, UB_SIZE};
    AscendC::LocalTensor<DataType> secondLocal_{AscendC::TPosition::VECIN, 0, UB_SIZE};
    AscendC::LocalTensor<DataType> resultLocal_{AscendC::TPosition::VECIN, 0, UB_SIZE};

    // ---- [USER] GM Tensors ----
    AscendC::GlobalTensor<DataType> outputGlobal_;
    AscendC::GlobalTensor<DataType> secondInputGlobal_;  // [USER] 无第二输入时删除

    int64_t stageSize_{0};
    ProblemShape problemShape_;

    // ============================================================
    // [PATTERN] Init — UB 分 stage + 绑定 GM
    // ============================================================
    __aicore__ inline void Init(
        Params const& params, int64_t baseM, int64_t baseN, ProblemShape& problemShape)
    {
        // [SAMPLE] ALIGN_ELEM = 32/sizeof(float) = 8；非 float 输出必须改
        constexpr int64_t ALIGN_ELEM = 32 / sizeof(DataType);
        int64_t baseNAlign = ::CeilDiv(baseN, ALIGN_ELEM) * ALIGN_ELEM;
        int64_t matmulArea = baseM * baseNAlign;

        // [USER] stageNum = UB 除 matmul 结果外还需要几个 stage 缓冲
        //   1 (Relu 无第二输入)             → resultLocal_ 独占剩余 UB
        //   2 (Div/Mul/Add 一路第二输入)    → secondLocal_ + resultLocal_   ← 本骨架默认
        //   3 (Mul+Add 二路第二输入)        → secondLocal_ + thirdLocal_ + resultLocal_
        constexpr int64_t stageNum = 2;   // [SAMPLE]
        int64_t lastUBBytes = UB_SIZE - matmulArea * sizeof(DataType);
        stageSize_ = AscendC::Std::min(
            static_cast<int64_t>(lastUBBytes / stageNum / sizeof(DataType) / baseNAlign * baseNAlign),
            matmulArea);

        int64_t ubOffset = matmulArea;
        secondLocal_ = cLocal_[ubOffset];      // [USER] 无第二输入时删除这两行
        ubOffset   += stageSize_;
        resultLocal_ = cLocal_[ubOffset];

        problemShape_ = problemShape;
        outputGlobal_.SetGlobalBuffer(
            reinterpret_cast<__gm__ DataType*>(params.outputGmAddr));
        secondInputGlobal_.SetGlobalBuffer(
            reinterpret_cast<__gm__ DataType*>(params.secondInputGmAddr));  // [USER]
    }

    // ============================================================
    // [PATTERN] GetTensor — 返回 matmul UB 目标（BlockMmad Fixpipe 写入此处）
    // ============================================================
    __aicore__ inline auto GetTensor() { return cLocal_; }

    // ============================================================
    // [PATTERN + USER] operator() — 每个 tile 调用
    // ============================================================
    __aicore__ inline void operator()(
        BlockShape const& blockShape, int64_t dstOffset,
        int64_t flagId = CvSync::AIV_TO_AIC_FLAG)
    {
        int64_t blockShapeM = Get<0>(blockShape);
        int64_t blockShapeN = Get<1>(blockShape);

        // [PATTERN] SPLIT_M：AIV0/AIV1 各吃一半 M（奇数 M 时 AIV0 多 1 行）
        int64_t halfM = ::CeilDiv(blockShapeM, AscendC::GetTaskRation());
        blockShapeM = ((static_cast<uint64_t>(blockShapeM) & 1UL) > 0UL)
                          ? (halfM - AscendC::GetSubBlockIdx()) : halfM;

        constexpr int64_t ALIGN_ELEM = 32 / sizeof(DataType);
        int64_t nAlign    = ::CeilDiv(blockShapeN, ALIGN_ELEM) * ALIGN_ELEM;
        int64_t inputSize = blockShapeM * nAlign;
        int64_t stageSize = AscendC::Std::min(stageSize_, inputSize) / nAlign * nAlign;
        int64_t N         = Get<MNK_N>(problemShape_);

        int64_t loop        = 0;
        int64_t stageOffset = 0;

        while (stageOffset < inputSize) {
            // [PATTERN] 当前 stage 写回 GM 的偏移（[SAMPLE] 输出 RowMajor）
            int64_t offset  = dstOffset + loop * stageSize / nAlign * N;
            offset         += AscendC::GetSubBlockIdx() * halfM * N;
            stageSize       = AscendC::Std::min(stageSize, inputSize - stageOffset);

            uint16_t nRows    = static_cast<uint16_t>(stageSize / nAlign);
            // [SAMPLE] sizeof(DataType) = sizeof(float) 时成立
            uint32_t rowBytes = static_cast<uint32_t>(blockShapeN * sizeof(DataType));
            uint32_t rowGap   = static_cast<uint32_t>((N - blockShapeN) * sizeof(DataType));

            // ---------- Step a: GM → secondLocal_ ([USER] 无第二输入可删) ----------
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ZERO_FLAG);

            AscendC::DataCopyExtParams   inCopyParams{nRows, rowBytes, rowGap, 0, 0};
            AscendC::DataCopyPadExtParams<DataType> inPadParams{false, 0, 0, 0};
            AscendC::DataCopyPad(secondLocal_, secondInputGlobal_[offset],
                                 inCopyParams, inPadParams);

            // ---------- Step b: 融合计算（[USER] 改动点）----------
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ZERO_FLAG);

            // [USER] 替换下面这一行为实际融合算子：
            //   Div : AscendC::Div(resultLocal_, cLocal_[stageOffset], secondLocal_, stageSize);
            //   Mul : AscendC::Mul(resultLocal_, cLocal_[stageOffset], secondLocal_, stageSize);
            //   Add : AscendC::Add(resultLocal_, cLocal_[stageOffset], secondLocal_, stageSize);
            //   Relu: AscendC::Relu(resultLocal_, cLocal_[stageOffset], stageSize);
            AscendC::Div(resultLocal_, cLocal_[stageOffset], secondLocal_, stageSize);
            AscendC::PipeBarrier<PIPE_V>();

            // ---------- Step c: resultLocal_ → GM ----------
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ZERO_FLAG);

            AscendC::DataCopyExtParams outCopyParams{nRows, rowBytes, 0, rowGap, 0};
            AscendC::DataCopyPad<DataType>(outputGlobal_[offset], resultLocal_, outCopyParams);

            stageOffset += stageSize;
            loop++;
        }

        // [PATTERN] 通知 AIC 本轮 AIV 完成
        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(flagId);
    }
};

#endif // EPILOGUE_SKELETON_H
