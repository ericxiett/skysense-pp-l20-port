# SkySense++ L20 torch2 重建 mIoU 坍缩根因诊断与修复报告（v1）

| 项目 | 内容 |
| --- | --- |
| 报告版本 | v1（修复已部署，最终 mIoU 待 Pod 收尾） |
| 目标 | 定位并修复 L20 节点（NVIDIA L20, sm_89）torch2 重建版 SkySense++ 在 flood3i 1-shot 推理中 mIoU 从 基线环境基准 0.5515 坍缩至 0.0089 的根因 |
| 根因 | `Proj_MHSA.forward` 改造时丢失 mmcv `MultiheadAttention` 包装类末尾的 `identity + dropout(proj_drop(out))` 残差连接，attn1/2/3 三处各少一个 `proj_out(x)` 项，致特征完全漂移 |
| 修复 | 远端 `~/SkySensePlusPlus/lib/models/backbones/swin_v2.py` 补回 `identity = x; x = x + identity`，已备份 `.bak-noresidual` |
| 验证 | 修复后推理早期 4 张样本的 pred（`save/vis_full/` 三联图）与 GT 几乎像素级一致；save 目录 2316 个 mtime>=08:00GMT 的 pred 文件中 <1KB（空白）仅 1.3% |

## 一、问题陈述

| 指标 | 数值 | 来源 |
| --- | --- | --- |
| 基线环境基准 mIoU | 0.5515 | 原始训练环境推理结果（mmcv MultiheadAttention 包 atorch FA） |
| L20 改造版 mIoU | 0.0089 | torch2.1 容器（mmcv 2.1.0），`Proj_MHSA` 改用 torch 原生 `nn.MultiheadAttention` |
| 现象 | 推理输出 pred 几乎全黑（842B 空白 PNG） | `_infer_check/pred_10169_*.png`（842B）与 GT `_infer_check/gt_10169_*.png` 对比 |

## 二、根因定位

### 2.1 基线版本 端 `Proj_MHSA` 实现（`_infer_check/baseline_orig/lib__models__backbones__swin_v2.py` 580-601 行）

```python
class Proj_MHSA(nn.Module):
    def __init__(self, embed_dims, proj_dims, num_heads=16, batch_first=True, bias=True):
        super().__init__()
        self.proj_in = nn.Linear(in_features=embed_dims, out_features=proj_dims)
        self.attn = MultiheadAttention(           # mmcv 包装
            embed_dims=proj_dims, num_heads=num_heads,
            batch_first=batch_first, bias=bias)
        self.proj_out = nn.Linear(in_features=proj_dims, out_features=embed_dims)
    def forward(self, x):
        x = self.proj_in(x)
        x = self.attn(x, x, x)                   # mmcv 包装返回 identity + attn_out
        x = self.proj_out(x)
        return x
```

`MultiheadAttention` 为 mmcv 包装类（`from mmcv.cnn.bricks.transformer import MultiheadAttention`）。mmcv 1.x/2.0/2.1 三个版本的 `MultiheadAttention.forward` 末尾均为 `return identity + self.dropout_layer(self.proj_drop(out))`，其中 `identity` 默认 `query` 即 `x`。底层是 atorch `MultiheadAttentionFA`（`attn.attn.Wqkv/out_proj` 键）。

### 2.2 L20 改造版 `Proj_MHSA` 实现（修复前，`_infer_check/l20_cur/lib__models__backbones__swin_v2.py` 582-614 行）

```python
class Proj_MHSA(nn.Module):
    def __init__(self, embed_dims, proj_dims, num_heads=16, batch_first=True, bias=True):
        super().__init__()
        self.proj_in = nn.Linear(in_features=embed_dims, out_features=proj_dims)
        self.attn = nn.MultiheadAttention(        # torch 原生，替换 mmcv 包装
            embed_dim=proj_dims, num_heads=num_heads,
            batch_first=batch_first, bias=bias)
        self.proj_out = nn.Linear(in_features=proj_dims, out_features=embed_dims)
    def forward(self, x):
        x = self.proj_in(x)
        x = self.attn(x, x, x, need_weights=False)[0]   # 仅 attn_out，缺 x 残差
        x = self.proj_out(x)
        return x
```

torch 原生 `nn.MultiheadAttention.forward` 不含 `identity` 残差语义，调用方须手动补回。

### 2.3 关键差异

| 端 | `Proj_MHSA(x)` 等效输出 | 残差项 |
| --- | --- | --- |
| 基线版本 | `proj_out( x + attn_out )` | 包含 `x` |
| L20 修复前 | `proj_out( attn_out )` | **缺失 `x`** |

attn1/2/3 三处（`embed_dims=352/704/1408`，`proj_dims=256/512/1024`）各缺一个 `proj_out(x)` 大项，三层累积致特征分布完全偏移，模型对所有像素预测为背景类，mIoU 坍缩至 0.0089。

## 三、证据链

### 3.1 排除项

- **方案 B：注意力 kernel 差异**：强制 torch2.1 SDPA flash-only backend 跑 1-shot，`kubectl logs <验证 Pod>` 尾部确认 `Mean 0.0089%`、逐类 IoU 与默认 SDPA **几乎逐位一致** → 基线环境 FA2 varlen kernel vs L20 SDPA 数值路径差异排除；
- **crop 一致性**：基线环境 pred 与 L20 pred 均为 512×512，`RandomResizedCrop(512, scale=(0.9999,1.0))` 两端一致；
- **eval pipeline**：输入 pipeline 是项目自定义 `pair_transforms`（ToTensor/RandomResizedCrop/Normalize），非 mmcv transforms，方案 C（mmcv transforms diff）适用场景排除；
- **s1/s2 vit encoder**：远端 `vit.py` 仍用 mmcv `MultiheadAttention` 包装（带残差），两端一致；eval config 仅 `backbone_hr.use_attn: True`、s1/s2 False → 改造差异只影响 hr 的 Proj_MHSA。

### 3.2 键加载 100% 正常（fullkeys2 实锤）

用远端真实 remap 逻辑（`_remap_proj_mhsa_keys` 只处理 hr attn1/2/3 精确前缀，s1/s2 保留双层由 load_model_weights 全局 `.Wqkv.->.in_proj_` 对齐）跑 `apply_fullkeys2.py` Pod：

```
===FK2_STATS===
FK2_MODELONLY count=0
MODELONLY_BY_PREFIX: {}
FK2_CKPTONLY count=0
FK2_COS
ATTN_KEYS_TOTAL=578 ATTN_BAD=0
FK2_NONATTN
NONATTN_KEYS_TOTAL=992 NONATTN_BAD=0
```

578 个注意力键 + 992 个非注意力键全部 cos 对齐。之前 fullkeys v1 的「288 个 model-only」是用旧版 remap 模拟的假象（`load4/load6` 历史日志已证伪：`dbg_load_check_v2.log` 显示「模型有但 new_dict 无（随机初始化）: 0」「共 578 个注意力键，异常 0 个」）。

## 四、修复

远端 `~/SkySensePlusPlus/lib/models/backbones/swin_v2.py` `Proj_MHSA.forward` 补回 mmcv 包装类残差（已备份 `.bak-noresidual`，`__pycache__` 已清，py_compile 语法校验通过）：

```python
def forward(self, x):
    x = self.proj_in(x)
    # [L20 patch 2.7 fix] 补回 mmcv MultiheadAttention 包装类的 identity 残差：
    # 基线版本 的 Proj_MHSA.attn 是 mmcv MultiheadAttention（包 atorch FA），
    # 其 forward 返回 identity + dropout(proj_drop(attn_out)) = x + attn_out；
    # L20 改用 torch 原生 MHA 后仅返回 attn_out，缺 x 残差 -> 特征漂移 ->
    # mIoU 坍缩(55%->0.9%)。此处手动补回，保持与训练一致。
    identity = x
    x = self.attn(x, x, x, need_weights=False)[0]
    x = x + identity
    x = self.proj_out(x)
    return x
```

## 五、修复效果

### 5.1 三联图（input + pred + gt）—— 4 张样本验证

推理早期（`~/SkySensePlusPlus/eval/flood3i_1shot/save/vis_full/`，mtime >= 08:00 GMT）拉取的 4 张样本，三联图均显示 pred 与 gt 几乎像素级一致：

| 样本 | 任务 | pred 与 gt 吻合度 |
| --- | --- | --- |
| 10169_lab_1_1-2.png | 房屋建筑区识别 | 几乎像素级一致 |
| 10815_lab_1_0-2.png | 房屋建筑区识别 | 几乎像素级一致 |
| 10809_lab_2_1-4.png | 道路+房屋识别 | 几乎像素级一致 |
| 11484_lab_2_0-5.png | 水域/堤岸分割 | 几乎像素级一致 |

旧 0.0089 时期对应 pred（`_infer_check/pred_10169_*.png`，842B 空白 PNG）完全不识别内容。

### 5.2 pred 文件大小分布

| 时间窗口 | 总数 | < 1KB（空白） | >= 2KB（有效） |
| --- | --- | --- | --- |
| 修复后（mtime >= 08:00 GMT） | 2316 | 30（1.3%） | 2083（90%） |
| 修复前累积 | 17357 | 7642（44%） | 9286（53.5%） |

修复后 99% 的 pred 不再空白，pred 内容恢复（与 GT 空间分布吻合）。

## 六、最终验证结果（Pod 完整跑完 4919 batch，1h05m）

验证实例于 2026-09-01 完成全部 4919 batch 推理并输出最终评估：

| 类别 | IoU | Accuracy |
| --- | --- | --- |
| 2 | 0.6561 | 0.8501 |
| 5 | 0.5336 | 0.7267 |
| 6 | 0.6195 | 0.8633 |
| 4 | 0.4736 | 0.7406 |
| 7 | 0.3544 | 0.6856 |
| 8 | 0.2380 | 0.7483 |
| 9 | 0.7148 | 0.8242 |
| 3 | 0.5616 | 0.7303 |
| 1 | 0.8098 | 0.9143 |
| **Mean** | **0.5513** | **0.7871** |

**结论：L20 修复后 mIoU 0.5513 vs 基线环境基准 0.5515，完全恢复**（差异 0.0002 为浮点噪声级）。修复效果普适（4919 batch 全量评估，非抽样），逐类 IoU 分布与基准一致（类别 1 建筑最高 0.8098，类别 8 最低 0.2380）。

完整评估日志：`~/SkySensePlusPlus/logs/start_1shot_fix.log`（`python lib/evaluation/segm_eval_base.py` 段）。

## 七、相关文件

| 文件 | 说明 |
| --- | --- |
| `apply_fix.py` | 验证 Pod 提交脚本（L20 节点 1-shot 推理） |
| `apply_fullkeys.py` / `apply_fullkeys2.py` | 全键 diff 验证 Pod 提交脚本 |
| `_infer_check/l20_cur/lib__models__backbones__swin_v2.py` | 修复后远端 swin_v2.py 留档 |
| `_infer_check/cur_state/checkpoint.py` | 远端 checkpoint.py 留档（含 mmcv 包装类残差语义说明 docstring） |
| `_infer_check/baseline_orig/lib__models__backbones__swin_v2.py` | 基线原始版 swin_v2.py 留档 |
| `_infer_check/fix_pred/` | 修复后推理单 pred 图（裁剪下半部分） |
| `_infer_check/fix_vis/` | 修复后推理三联图（input + pred + gt） |
