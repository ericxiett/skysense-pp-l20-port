# SkySense++ L20 (torch2) 移植适配层

> SkySense++ 在 NVIDIA L20 (Ada / sm_89) + PyTorch 2.1 环境下的移植适配与关键修复。
> 上游项目：[kang-wu/SkySensePlusPlus](https://github.com/kang-wu/SkySensePlusPlus)（flood3i 1-shot 遥感分割）。

## 背景

上游 SkySensePlusPlus 依赖 torch 1.13 + cu117 + atorch 自研注意力实现，无法在 L20（sm_89, Ada 架构，torch2.1 + cu121 容器）上直接运行。本仓库是**适配层**（非完整项目镜像）：包含 17 个源文件的适配 diff、关键修复文件的完整版本、L20 运行脚本与部署文档。

**核心修复**：`Proj_MHSA` 从 mmcv `MultiheadAttention`（包 atorch FA，forward 带 identity 残差）改造为 torch 原生 `nn.MultiheadAttention` 时丢失了残差连接，导致推理 mIoU 从 NFS 7/20 基准 **0.5515** 坍缩至 **0.0089**；补回 `x = x + identity` 后恢复至 **0.5513**，与基准完全对齐。

## 目录结构

```
skysense-pp-l20-port/
├── patches/
│   └── skysense_pp_l20_port.patch    # 17 文件完整 diff（基于上游 cb0c6b7）
├── lib/                              # 关键适配文件完整版本（修复后）
│   ├── models/backbones/swin_v2.py   # Proj_MHSA 残差修复（根因修复点）
│   ├── utils/checkpoint.py           # Wqkv->in_proj_ remap（只处理 hr attn1/2/3）
│   ├── predictors/flood3i_1shot.py   # 1-shot 预测器
│   └── datasets/                     # loader / transforms L20 适配
├── configs/
│   └── eval_skysense_pp_flood3i.yml  # replace_speedup_op: False 等
├── scripts/
│   ├── start_1shot.sh                # 容器内安装+运行脚本（1-shot 全流程）
│   ├── start_dbg.sh                  # 调试诊断脚本
│   ├── run_1shot.sh                  # 推理入口
│   ├── requirements_l20.txt          # L20 依赖清单（mmcv 2.1.0 + torch2.1）
│   └── Dockerfile                    # 基础镜像定义
├── docs/
│   ├── 部署文档_v1.md                # 部署介质使用说明（跳板机分发）
│   └── SkySense++_L20_Proj_MHSA_根因诊断与修复报告_v1.1.md
└── deploy/                           # 部署介质包（tar.gz 内容同源）
```

## 快速开始

1. 从上游克隆完整项目：`git clone https://github.com/kang-wu/SkySensePlusPlus.git`
2. 应用适配补丁：`git apply patches/skysense_pp_l20_port.patch`
3. 按 `docs/部署文档_v1.md` 配置环境（基础镜像已推送至本环境 Harbor `<Harbor 地址>:5000/library/pytorch:2.1.2-cuda12.1-cudnn8-devel`，依赖见 `scripts/requirements_l20.txt`）
4. 执行 1-shot 推理：`bash scripts/start_1shot.sh`

详细部署步骤、依赖安装、数据放置、已知问题见部署文档。

## 适配要点摘要

| 模块 | 适配内容 |
| --- | --- |
| `lib/models/backbones/swin_v2.py` | `Proj_MHSA` 改用 torch 原生 MHA + 补回 mmcv 包装类 identity 残差（0.0089 → 0.5513） |
| `lib/utils/checkpoint.py` | `_remap_proj_mhsa_keys`：ckpt `Wqkv/out_proj` 键 → torch `in_proj_/out_proj`（只处理 hr attn1/2/3，s1/s2 保留双层结构靠循环改名对齐） |
| `lib/datasets/loader/*` | crop 512 对齐、Gdal 依赖处理 |
| `lib/datasets/utils/transforms.py` | torchvision 0.16 兼容（antialias 等） |
| `configs/eval_skysense_pp_flood3i.yml` | `replace_speedup_op: False`（L20 无 atorch 加速算子） |
| `scripts/requirements_l20.txt` | mmcv==2.1.0 / mmsegmentation==1.2.2 / mmengine，走 openmmlab cu121-torch2.1 预编译 wheel |

## 验证结果（flood3i 1-shot, 4919 batch 全量）

| 指标 | NFS 7/20 基准 | L20 修复前 | L20 修复后 |
| --- | --- | --- | --- |
| Mean mIoU | 0.5515 | 0.0089 | **0.5513** |
| Mean Acc | 0.7871 | — | **0.7871** |

## 说明

- 本仓库仅包含适配层与文档，不含数据集、模型权重与运行日志；
- 推理环境、数据路径等按部署文档配置；不同硬件需按需调整；
- 上游协议与版权归原项目所有，本仓库适配层按相同协议分发。
