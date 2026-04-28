// ============================================================
// DivEpilogue — AIV 侧 Div 融合 Epilogue（参考实现 / 主样板）
//
// [CODE LINEAGE]
//   本文件是 **mxfp8 + Div** 这个具体算子组合的完整可运行参考实现。
//   开发其他 mxfp8 + eltwise 算子时，**复制本文件 → 按下面三类标注定向修改**，
//   不要整份照抄然后在无关位置随意增删。
//
//   标注分层：
//     [PATTERN]  融合 Epilogue 必备骨架 — 任何算子都保留
//     [USER]     业务语义变量 — 必须替换为你的算子语义
//     [SAMPLE]   Div/float 样例恰好成立的取值 — 其他算子必须重新评估
//
// 三接口模式（[PATTERN]）:
//   Init(params, l1M, l1N, problemShape)    — 划分 UB 布局
//   GetTensor()                             — 返回 cLocal_（供 AIC Fixpipe 目标）
//   operator()(blockShape, offsetC, flagId) — 计算 + 写出 GM + CV SetFlag
//
// UB 布局（本样例固定为 2 stage，[SAMPLE]）:
//   [0, l1M * l1NAlign)                   : matmul 结果（cLocal_，offset 固定 0）
//   [ubOffset,        +stageSize_)        : 第二路输入缓冲（dLocal_）
//   [ubOffset+stageSize_, +stageSize_)    : 计算结果缓冲（cLocalTmp_）
// ============================================================
#ifndef DIV_EPILOGUE_H
#define DIV_EPILOGUE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "epilogue/cv_sync_constants.h"
#include "utils/hardware_constants.h"
#include "kernel_utils/common_utils.h"

using namespace AscendC;

class DivEpilogue {
public:
    static constexpr uint16_t ZERO_FLAG = 0;
    static constexpr uint16_t AIC_SYNC_AIV_MODE_4 = CvSync::MODE;

    // [PATTERN] UB 物理大小（从 hardware_constants.h 获取，避免跨文件硬编码）
    static constexpr uint32_t UB_SIZE = ::Hardware::UB_SIZE;

    // [SAMPLE] 当前样例的计算精度固定 float：matmul 结果、第二路输入、输出同精度。
    //          若要做 half/bf16 输出，需把 DataType 换成对应类型，并相应修改：
    //            - ALIGN_ELEM = 32 / sizeof(OutputType)
    //            - GlobalTensor<DataType> 模板参数
    //            - rowBytes/srcGap 中的 sizeof(float) → sizeof(DataType)
    using DataType = float;

    // [USER] Arguments / Params 字段名与 Host 侧 Kernel Entry 签名对齐
    struct Arguments {
        GM_ADDR divisorGmAddr{nullptr};   // [USER] 第二路输入 GM 地址（Relu 等单输入算子可删除此字段）
        GM_ADDR outputGmAddr{nullptr};
    };

    struct Params {
        GM_ADDR divisorGmAddr{nullptr};
        GM_ADDR outputGmAddr{nullptr};
    };

    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

    // [PATTERN] cLocal_ 位于 UB offset 0（与 BlockMmad Fixpipe 目标固定对齐）
    AscendC::LocalTensor<DataType> cLocal_{AscendC::TPosition::VECIN, 0, UB_SIZE};
    // [SAMPLE] 2 stage 布局：计算结果暂存 + 第二路输入暂存
    AscendC::LocalTensor<DataType> cLocalTmp_{AscendC::TPosition::VECIN, 0, UB_SIZE};
    AscendC::LocalTensor<DataType> dLocal_{AscendC::TPosition::VECIN, 0, UB_SIZE};

    // [USER] GM tensors：dtype = OutputType（当前样例全 float）
    AscendC::GlobalTensor<DataType> outputGlobal_;
    AscendC::GlobalTensor<DataType> divisorGlobal_;

    int64_t stageSize_{0};
    ProblemShape problemShape_;

    // ---- [PATTERN] Init ----
    __aicore__ inline void Init(
        Params const& params, int64_t l1M, int64_t l1N, ProblemShape& problemShape)
    {
        // [SAMPLE] ALIGN_ELEM 按 float = 8；非 float 输出时必须改为 32/sizeof(OutputType)
        constexpr int64_t ALIGN_ELEM = 32 / sizeof(DataType);
        int64_t l1NAlign = ::CeilDiv(l1N, ALIGN_ELEM) * ALIGN_ELEM;
        int64_t matmulArea = l1M * l1NAlign;

        // [USER] stageNum = 业务需要的 UB 分段数
        //   - 1（Relu 等无额外输入）：剩余 UB 全部给 cLocalTmp_
        //   - 2（Div/Mul/Add 一路第二输入）：当前样例
        //   - 3（Mul+Add 二路第二输入）：dLocal_ × 2 + cLocalTmp_
        constexpr int64_t stageNum = 2;   // [SAMPLE] 当前样例为双输入融合（Matmul 输出 + divisor）
        int64_t lastUBBytes = UB_SIZE - matmulArea * sizeof(DataType);
        stageSize_ = AscendC::Std::min(
            static_cast<int64_t>(lastUBBytes / stageNum / sizeof(DataType) / l1NAlign * l1NAlign),
            matmulArea);

        int64_t ubOffset = matmulArea;
        dLocal_ = cLocal_[ubOffset];            // [USER] 第二路输入缓冲起点
        ubOffset += stageSize_;
        cLocalTmp_ = cLocal_[ubOffset];         // [PATTERN] 计算结果缓冲起点

        problemShape_ = problemShape;
        outputGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType*>(params.outputGmAddr));
        divisorGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType*>(params.divisorGmAddr));
    }

    // ---- [PATTERN] GetTensor：BlockMmad Fixpipe 写入目标 ----
    __aicore__ inline auto GetTensor() { return cLocal_; }

    // ---- [PATTERN] operator()：计算 + 写出 + CV SetFlag ----
    __aicore__ inline void operator()(
        BlockShape const& blockShape, int64_t dstOffset, int64_t flagId = CvSync::AIV_TO_AIC_FLAG)
    {
        int64_t blockShapeM = Get<0>(blockShape);
        int64_t blockShapeN = Get<1>(blockShape);

        // [PATTERN] SPLIT_M：AIV0/AIV1 各处理一半 M，奇数 M 由 SubBlockIdx 修正
        int64_t halfM = ::CeilDiv(blockShapeM, AscendC::GetTaskRation());
        blockShapeM = ((static_cast<uint64_t>(blockShapeM) & 1UL) > 0UL)
                          ? (halfM - AscendC::GetSubBlockIdx()) : halfM;

        constexpr int64_t ALIGN_ELEM = 32 / sizeof(DataType);
        int64_t nAlign = ::CeilDiv(blockShapeN, ALIGN_ELEM) * ALIGN_ELEM;
        int64_t inputSize = blockShapeM * nAlign;
        int64_t stageSize = AscendC::Std::min(stageSize_, inputSize) / nAlign * nAlign;
        int64_t N = Get<MNK_N>(problemShape_);   // [PATTERN] 全局 N，用于 stride

        int64_t loop = 0;
        int64_t stageOffset = 0;

        while (stageOffset < inputSize) {
            // [PATTERN] 按 RowMajor 累加行偏移 + SPLIT_M 偏移
            //           [SAMPLE] dstOffset = mPos * N + nPos 假设**输出为 RowMajor**
            int64_t offset = dstOffset + loop * stageSize / nAlign * N;
            offset += AscendC::GetSubBlockIdx() * halfM * N;
            stageSize = AscendC::Std::min(stageSize, inputSize - stageOffset);

            // [PATTERN] Step a: GM → UB 搬第二路输入（MTE2，带 stride）
            // [USER]    若你的算子无第二路输入（如 Relu/GeLU/Cast），删除本 Step
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ZERO_FLAG);

            uint16_t nRows = static_cast<uint16_t>(stageSize / nAlign);
            // [SAMPLE] sizeof(float) —— 输出/第二路输入 dtype 改变时必须同步修改
            uint32_t rowBytes = static_cast<uint32_t>(blockShapeN * sizeof(DataType));
            uint32_t srcGap = static_cast<uint32_t>((N - blockShapeN) * sizeof(DataType));
            AscendC::DataCopyExtParams dCopyParams{nRows, rowBytes, srcGap, 0, 0};
            AscendC::DataCopyPadExtParams<DataType> dPadParams{false, 0, 0, 0};
            AscendC::DataCopyPad(dLocal_, divisorGlobal_[offset], dCopyParams, dPadParams);

            // [PATTERN] Step b: MTE2 → V 同步
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ZERO_FLAG);

            // [USER] Step c: 业务计算。替换这一行即可切换 eltwise 类型。
            //   [SAMPLE-alt: Div]       AscendC::Div(cLocalTmp_, cLocal_[stageOffset], dLocal_, stageSize);   // 当前样板
            //   [SAMPLE-alt: Mul]       AscendC::Mul(cLocalTmp_, cLocal_[stageOffset], dLocal_, stageSize);
            //   [SAMPLE-alt: Add]       AscendC::Add(cLocalTmp_, cLocal_[stageOffset], dLocal_, stageSize);
            //   [SAMPLE-alt: Relu]      AscendC::Relu(cLocalTmp_, cLocal_[stageOffset], stageSize);           // 无第二路输入：删 Step a 与 dLocal_；stageNum=1
            //   [SAMPLE-alt: Cast->half] AscendC::Cast(castLocal_, cLocal_[stageOffset], CAST_RINT, stageSize); // 输出 dtype 变：同步改 using DataType / sizeof / GlobalTensor 模板；详见 matmul_fusion_guide.md“附录 B：派生算子改动矩阵”
            //   [SAMPLE-alt: Div+Relu]  先 Div → PipeBarrier<PIPE_V> → Relu(cLocalTmp_, cLocalTmp_, ...)      // UB 布局同 Div
            AscendC::Div(cLocalTmp_, cLocal_[stageOffset], dLocal_, stageSize);
            AscendC::PipeBarrier<PIPE_V>();

            // [PATTERN] Step d: V → MTE3 同步 + 写回 GM（带 stride）
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ZERO_FLAG);

            AscendC::DataCopyExtParams outParams{nRows, rowBytes, 0, srcGap, 0};
            AscendC::DataCopyPad<DataType>(outputGlobal_[offset], cLocalTmp_, outParams);

            stageOffset += stageSize;
            loop++;
        }

        // [PATTERN] CV 同步：通知 AIC 本 tile 的 UB 已释放
        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(flagId);
    }

    // [PATTERN] Arguments → Params 转换（Host 侧调用）
    __host_aicore__ static Params InitParams(Arguments const& args)
    {
        return {args.divisorGmAddr, args.outputGmAddr};
    }
};

#endif // DIV_EPILOGUE_H
