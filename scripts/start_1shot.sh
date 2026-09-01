#!/bin/bash
# ============================================================
# start_1shot.sh —— SkySense++ flood3i 1-shot 容器内"安装+运行"脚本
# 挂载后容器内路径：/workspace/start_1shot.sh（hostPath ~/SkySensePlusPlus）
# 基础镜像：<镜像仓库地址>:5000/library/pytorch:2.1.2-cuda12.1-cudnn8-devel
#           （已推送至私有镜像仓库，docker.io 源不可直连；已预装 torch2.1.2+cu121）
# 2026-08-31 实测修正（相对手册原草案）：
#   [1] antmmf 无 setup.py，pip install -e 必失败；run_1shot.sh 内建
#       `cd antmmf && export PYTHONPATH=$(pwd)`，靠 PYTHONPATH 引用，不 pip 安装
#   [2] run_1shot.sh 内部用 `python` 命令，容器若只有 python3 需补软链
#   [3] mmcv 走 openmmlab cu121/torch2.1 预编译 wheel（L20 节点 已实测 HTTP 200）
#   [4] pip 增强：PIP_CACHE_DIR 指向 /workspace/.pip-cache（hostPath 持久化，跨 Pod
#       复用下载缓存，mmcv 94MB 不用重下）；--timeout 60 --retries 5 防下载超时；
#       失败自动重试 3 次
#   [5] 2026-08-31 实测：清华 PyPI 源对 L20 节点 卡死（20s 无响应，连续 Read timeout，
#       甚至导致 wheel 截断 hash 校验失败）；阿里云源 4.5MB/s、腾讯云 1.6MB/s。
#       -> PIP_INDEX_URL 改用阿里云（mmcv 仍走 openmmlab --find-links，不受影响）
# ============================================================
set -e
export DEBIAN_FRONTEND=noninteractive
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
export PIP_TRUSTED_HOST=mirrors.aliyun.com
export PIP_CACHE_DIR=/workspace/.pip-cache
mkdir -p "$PIP_CACHE_DIR"

cd /workspace

echo "==> [0/3] 环境检查"
python3 -V
python3 -c "import torch; print('torch', torch.__version__, '| cuda_avail', torch.cuda.is_available(), '| dev', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
# run_1shot.sh 内部用 `python`；容器通常只有 python3，补一个软链
if ! command -v python >/dev/null 2>&1; then
  echo "==> 未发现 python 命令，ln -sf python3 -> /usr/local/bin/python"
  ln -sf "$(command -v python3)" /usr/local/bin/python
fi

echo "==> [1/3] 安装 torch2 适配依赖（首次约 10-20 分钟，失败最多重试 3 次）..."
for attempt in 1 2 3; do
  echo "---- pip 安装尝试 #$attempt/3 ----"
  if pip install --timeout 60 --retries 5 -r requirements_l20.txt; then
    echo "==> 依赖安装成功"
    break
  else
    echo "==> 依赖安装失败（尝试 #$attempt/3），5 秒后重试..."
    sleep 5
    if [ "$attempt" = "3" ]; then
      echo "!! 3 次尝试均失败，退出"
      exit 1
    fi
  fi
done

echo "==> [2/3] antmmf 运行时依赖 + import 冒烟测试"
# antmmf 无 setup.py，走 PYTHONPATH；但 __init__.py 全量子模块导入，
# 按全包 import 清单补充运行时依赖（transformers 等），并做冒烟测试
# 注：torchtext 版本须与 torch 2.1.2 匹配（0.16.x）；
#     transformers 须 <4.50（4.57 要求 torch>=2.3 的 register_pytree_node API，torch 2.1.2 没有）
ls -l antmmf/antmmf/__init__.py
for attempt in 1 2 3; do
  echo "---- antmmf 依赖安装尝试 #$attempt/3 ----"
  if pip install --timeout 60 --retries 5 \
      "transformers==4.44.2" sentencepiece omegaconf regex deprecated \
      jsonlines lmdb nltk rouge tensorboardX "torchtext==0.16.2" \
      pycocotools decord "mmcls==1.0.0rc6" ftfy; then
    echo "==> antmmf 依赖安装成功"
    break
  else
    echo "==> antmmf 依赖安装失败（尝试 #$attempt/3），5 秒后重试..."
    sleep 5
    if [ "$attempt" = "3" ]; then
      echo "!! 3 次尝试均失败，退出"
      exit 1
    fi
  fi
done
# ---- mmcls / mmseg 兼容处理（swin_v2.py / vit.py 依赖）----
# [1] mmcls 1.0.0rc6 断言 mmcv<2.1.0，但 mmcv 2.1.0 API 兼容，放宽上限
#     注意：不能用 `python3 -c "import mmcls"` 定位（import 会触发 mmcv->cv2->libGL 崩溃），
#     直接按已知 site-packages 路径处理
MMCLS_DIR="/opt/conda/lib/python3.10/site-packages/mmcls"
if [ -f "$MMCLS_DIR/__init__.py" ]; then
  # [L20 patch 2.6 fix] 原 sed 精确匹配 `mmcv_maximum_version = '2.1.0'` 曾因
  # wheel 内格式差异静默失效（断言仍报 <2.1.0），改 Python regex 幂等处理
  python3 - <<'PYEOF'
import pathlib, re
f = pathlib.Path("/opt/conda/lib/python3.10/site-packages/mmcls/__init__.py")
s = f.read_text()
s2 = re.sub(r"mmcv_maximum_version\s*=\s*['\"][^'\"]+['\"]",
            "mmcv_maximum_version = '2.3.0'", s)
if s2 != s:
    f.write_text(s2)
    print("==> mmcls 版本上限已放宽到 2.3.0 (regex)")
else:
    cur = re.findall(r"mmcv_maximum_version\s*=\s*['\"][^'\"]+['\"]", s)
    print(f"==> mmcls 当前上限: {cur or '未匹配到'}")
PYEOF
  # [L20 patch 2.6 fix] 清除 mmcls 包 __pycache__：regex 改的是 .py 源文件，
  # 但旧 .pyc 字节码（编译时上限还是 2.1.0）若 mtime/size 校验"碰巧通过"
  # （2.1.0->2.3.0 长度相同），import 会复用旧字节码 -> 断言仍报 <2.1.0。
  find "$MMCLS_DIR" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
  echo "==> mmcls __pycache__ 已清除"
  # [2] swin_v2.py 用 mmcls.utils.get_root_logger（旧 API，1.x 已移除）——注入兼容实现
  MMCLS_UTILS="$MMCLS_DIR/utils/__init__.py"
  if ! grep -q 'get_root_logger' "$MMCLS_UTILS"; then
    cat >> "$MMCLS_UTILS" <<'PYEOF'

# [L20 patch 2.6] 兼容 swin_v2.py 旧 API：mmcls 1.x 移除 get_root_logger
from mmengine.logging import MMLogger


def get_root_logger():
    return MMLogger.get_current_instance()
PYEOF
    echo "==> mmcls.utils 已注入 get_root_logger 兼容实现"
    # 注入的 utils/__init__.py 同样可能有旧 pyc，一并清除
    find "$MMCLS_DIR" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
    echo "==> mmcls __pycache__ 已二次清除"
  else
    echo "==> mmcls.utils 已有 get_root_logger，跳过"
  fi
else
  echo "!! WARN: 未找到 mmcls site-packages，跳过兼容处理"
fi
# [3] vit.py / up_head.py 用 mmseg.utils.get_root_logger（0.x API，1.x 已移除）——同样注入
MMSEG_UTILS="/opt/conda/lib/python3.10/site-packages/mmseg/utils/__init__.py"
if [ -f "$MMSEG_UTILS" ] && ! grep -q 'get_root_logger' "$MMSEG_UTILS"; then
  cat >> "$MMSEG_UTILS" <<'PYEOF'

# [L20 patch 2.6] 兼容 vit.py/up_head.py 旧 API：mmseg 1.x 移除 get_root_logger
from mmengine.logging import MMLogger


def get_root_logger():
    return MMLogger.get_current_instance()
PYEOF
  echo "==> mmseg.utils 已注入 get_root_logger 兼容实现"
  # 清除 mmseg 包 __pycache__，防止旧字节码复用
  find "/opt/conda/lib/python3.10/site-packages/mmseg" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
  echo "==> mmseg __pycache__ 已清除"
fi
# mmcv 依赖装的是 opencv-python（非 headless），其 cv2 需要系统 libGL.so.1（容器没有）；
# 卸载之并强制重装 opencv-python-headless（无需 libGL，功能等价）恢复 cv2
pip uninstall -y opencv-python 2>/dev/null || echo "WARN: opencv-python 卸载未执行"
pip install --force-reinstall --no-deps opencv-python-headless 2>&1 | tail -2
# osgeo(gdal)：容器无系统 libgdal，pip 版需源码编译。已对 pretraining_loader.py
# 做惰性 import patch（宿主机持久化，见 .bak-l20-gdal），1-shot 链不触碰，无需安装。
# 冒烟：用 predictor 真实 import 路径（PYTHONPATH 同 run_1shot.sh）
cd /workspace && PYTHONPATH=/workspace/antmmf python3 -c "import antmmf.common.registry; print('antmmf registry import ok')" || echo "!! WARN: antmmf import 仍有问题，见上方报错"

echo "==> [3/3] 运行 flood3i 1-shot（GPU 0，seed 0）..."
# 注意：dataset 参数必须是 flood3i（无连字符）——run_1shot.sh 据此拼出
#   configs/eval_skysense_pp_flood3i.yml（即步骤 2.4 patch 的文件）与
#   lib/predictors/flood3i_1shot.py。曾误用 flood-3i 导致文件找不到。
# run_1shot.sh 是 bash 脚本，必须用 bash 执行（曾误用 python3 触发 SyntaxError）；
# 其内部 `python lib/predictors/...` 依赖 [0/3] 已做的 python3 软链
bash tools/run_1shot.sh 0 flood3i

echo "==> [done] start_1shot.sh 执行完毕"
