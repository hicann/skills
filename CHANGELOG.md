## 🔥 更新日志

### 【2026-04-14】
#### 新特性 New Features
- 【仿真，ops-simulator】新增Ascend 950 仿真Skill：支持 Ascend 950 仿真，并且输出性能分析报告和流水线图。
#### 特性增强 Feature Enhancement
- 【UT单元测试，ascendc-ut-develop】支持针对 ops-transformer 算子仓的 UT、CSV 代码重构，分离数据与测试代码，提高调试效率。
- 【aclnn 接口测试用例设计，ascendc-st-design】新增支持aclIntArray / aclFloatArray / aclBoolArray / aclScalarList / aclIntArray类型接口生成ST用例。
#### 问题修复 Bug Fix
- 【算子直调，ops-direct-invoke】修改verify_environment.sh脚本，返回environment.json 固定为1的bug，应该按实际设备的npu count返回。
    

### 【2026-04-13】
#### 新特性 New Features
- 【Team调度】支持team级代码条例全量检视，review team 派发条例给代码检视。支持子agent 并行检视、验证，提升检视效果，降低上下文的压力。


### 【2026-04-10】
#### 新特性 New Features
- 【PyPTO】新增 Skill：pypto-api-explore，PyPTO API 探索与文档查阅。
- 【PyPTO】新增 Skill：pypto-golden-generate，Golden 数据生成与验证。
- 【PyPTO】新增 Skill：pypto-intent-understand，用户需求解析与规格生成。
- 【PyPTO】新增 Skill：pypto-op-design 算子方案设计，含快速参考和设计模板。
- 【PyPTO】新增 Skill：pypto-op-develop 算子开发实现，含错误排查、约束参考、测试模板和环境脚本。
- 【PyPTO】新增 Skill：pypto-op-perf-tune 性能分析与调优（frontend / incore / swimlane 三个子模块）。
- 【PyPTO】新增 Skill：pypto-precision-debug 精度问题定位与调试。
- 【PyPTO】新增 Skill：pypto-precision-compare 精度对比验证（含二分查找和自动化脚本）。
- 【PyPTO】新增 Agent：pypto-op-analyst: 算子分析 Agent
- 【PyPTO】新增 Agent：pypto-op-developer: 算子开发 Agent
- 【PyPTO】新增 Agent：pypto-op-perf-tuner: 性能调优 Agent
- 【PyPTO】新增 Team：pypto-op-orchestrator 算子开发编排 （含初始化脚本和快速入门）。


### 【2026-04-09】
#### 特性增强 Feature Enhancement
 - 【Ascend C】【代码检视，ascendc-ops-reviewer】优化了 ascendc-ops-reviewer agent 的检视流程，增加了多维度检视表格、代码侧别识别和置信度评定机制。新增 Ascend C 的 API 最佳实践、性能编码规范和 TopK 编码问题清单。


### 【2026-04-07】
#### 新特性 New Features
- 【Ascend C】【代码检视，ascendc-ops-reviewer】ascendc-ops-reviewer agent支持GitCode PR的代码检视。
- 【Ascend C】【Kernel 架构】新增 Agent：ascendc-kernel-architect，<<<>>>直调支持多agent协同。
- 【Ascend C】【Kernel 架构】新增 Skill：ascendc-direct-invoke-template。
 
### 【2026-04-02】
#### 文档 Documentation
- 【开发规范】新增  CANNBot 开发规范，包含：Skill、Agents、Teams。
 #### 配置 Configuration
- 【Issue模板】新增 Issue 模板。


### 【2026-04-01】
#### 特性增强 Feature Enhancement
- 【Ascend C】【代码检视，ascendc-ops-reviewer】ascendc-ops-reviewer agent支持GitCode PR的代码检视。


### 【2026-03-26】
#### 新特性 New Features
- 【Ascend C】【白盒测试用例，ascendc-whitebox-design】新建Skill：ascendc-whitebox-design，白盒测试用例。


### 【2026-03-25】
#### 特性增强 Feature Enhancement
- 【Ascend C】【代码检视，ascendc-ops-reviewer】搭建code reviewer agent基础框架和工作流、支持检视条款的扩充和修改。


### 【2026-03-20】
#### 新特性 New Features
- 【Ascend C】新增Skill：ascendc-api-best-practices，Ascend C 的 API 使用最佳实践。
- 【Ascend C】新增Skill：ascendc-code-review，Ascend C 代码检视。
- 【Ascend C】新增Skill：ascendc-docs-search，文档搜索。
- 【Ascend C】新增Skill：ascendc-env-check，NPU 设备查询、CANN 环境验证。
- 【Ascend C】新增Skill：ascendc-kernel-develop-workflow，七阶段工作流。
- 【Ascend C】新增Skill：ascendc-npu-arch，NPU 架构知识、芯片型号映射。
- 【Ascend C】新增Skill：ascendc-precision-debug，算子精度调试。
- 【Ascend C】新增Skill：ascendc-runtime-debug，算子运行时错误调试。
- 【Ascend C】新增Skill：ascendc-st-design，接口测试用例设计。
- 【Ascend C】新增Skill：ascendc-task-focus，任务聚焦，解决长任务“迷失在中间”的问题。
- 【Ascend C】新增Skill：ascendc-tiling-design，Tiling 设计方法论。
- 【Ascend C】新增Skill：ascendc-ut-develop。UT单元测试用例开发与覆盖率增强。