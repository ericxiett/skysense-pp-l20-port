#!/bin/bash
# ============================================================
# start_dbg.sh —— 单样本 debug：复用 start_1shot.sh 的依赖安装段
# [0/3]-[2/3] 完全相同；[3/3] 改为跑 dbg_keys.py（不做 4919 全量推理）
# 日志 tee 到 /workspace/logs/dbg_keys.log（hostPath 持久化）
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
if ! command -v python >/dev/null 2>&1; then
  echo "==> 未发现 python 命令，ln -sf python3 -> /usr/local/bin/python"
  ln -sf "$(command -v python3)" /usr/local/bin/python
fi

echo "==> [1/3] 安装 torch2 适配依赖（.pip-cache 有缓存，应较快）..."
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

echo "==> [2/3] antmmf 运行时依赖 + 兼容处理"
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
  # 但旧 .pyc 字节码（含 mmcv_maximum_version='2.1.0' 的编译结果）若仍有效
  # （2.1.0→2.3.0 长度相同，pyc 头部源 mtime/size 校验会误判缓存有效），
  # import 时会复用旧字节码导致断言仍报 <2.1.0。删掉强制下次重新编译。
  find "$MMCLS_DIR" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
  echo "==> mmcls __pycache__ 已清除"
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
  fi
else
  echo "!! WARN: 未找到 mmcls site-packages，跳过兼容处理"
fi
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
pip uninstall -y opencv-python 2>/dev/null || echo "WARN: opencv-python 卸载未执行"
pip install --force-reinstall --no-deps opencv-python-headless 2>&1 | tail -2
cd /workspace && PYTHONPATH=/workspace/antmmf python3 -c "import antmmf.common.registry; print('antmmf registry import ok')" || echo "!! WARN: antmmf import 仍有问题，见上方报错"

echo "==> [3/3] 运行 dbg_keys.py（单样本 debug）..."
mkdir -p logs
python3 dbg_keys.py 2>&1 | tee logs/dbg_keys.log
echo "==> [done] start_dbg.sh 执行完毕"
