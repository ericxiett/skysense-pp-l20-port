# Copyright (c) Ant Financial Service Group. and its affiliates.
import os
import warnings

import torch

from antmmf.common import constants
from antmmf.common.registry import registry
from antmmf.common.checkpoint import Checkpoint
from antmmf.utils.distributed_utils import is_main_process

class SegCheckpoint(Checkpoint):
    def __init__(self, trainer, load_only=False):
        super().__init__(trainer, load_only=False)

    @staticmethod
    def _remap_proj_mhsa_keys(ckpt_model):
        """[L20 patch 2.6 fix] Proj_MHSA key 对齐（v1 回归）。

        训练时 Proj_MHSA.attn 是 mmcv MultiheadAttention 包 atorch
        MultiheadAttentionFA，state_dict key 为:
            backbone_hr.attn{1,2,3}.attn.attn.Wqkv.weight / .out_proj.weight
        L20 重建无 atorch，Proj_MHSA 改用 torch 原生 nn.MultiheadAttention
        (need_weights=False 走 SDPA 防 OOM)，key 为:
            backbone_hr.attn{1,2,3}.attn.in_proj_weight / .attn.out_proj.weight
        这里把 ckpt 的 Proj_MHSA 相关 key 重映射到新结构，否则 hr 的注意力
        权重会被 _load_state_dict 静默跳过 -> 随机初始化 -> mIoU 坍缩(55%->1.3%)。

        注意：s2/s1 的 vit encoder 不需要在此处理——其 attn 为
        mmcv MultiheadAttention 包 torch nn.MultiheadAttention，结构与 ckpt
        (mmcv MHA 包 atorch FA) 一致（均为 attn.attn 双层），由
        load_model_weights 循环中的全局 .Wqkv.->.in_proj_ 改名对齐。
        """
        out = {}
        for k, v in ckpt_model.items():
            nk = k
            for a in ("attn1", "attn2", "attn3"):
                p = f"backbone_hr.{a}.attn.attn."
                if k.startswith(p):
                    nk = k.replace(p, f"backbone_hr.{a}.attn.")
                    nk = nk.replace(".Wqkv.", ".in_proj_")
                    break
            out[nk] = v
        return out

    def load_model_weights(self, file, force=False):
        self.trainer.writer.write("Loading checkpoint")
        ckpt = self._torch_load(file)
        if registry.get(constants.STATE) is constants.STATE_ONLINE_SERVING:
            data_parallel = False
        else:
            data_parallel = registry.get("data_parallel") or registry.get(
                "distributed")

        if "model" in ckpt:
            ckpt_model = ckpt["model"]
        else:
            ckpt_model = ckpt
            ckpt = {"model": ckpt}

        # Proj_MHSA key 对齐（见 _remap_proj_mhsa_keys docstring）
        ckpt_model = self._remap_proj_mhsa_keys(ckpt_model)

        new_dict = {}

        # TODO: Move to separate function
        for attr in ckpt_model:
            if "fa_history" in attr:
                new_dict[attr.replace("fa_history",
                                      "fa_context")] = ckpt_model[attr]
            elif data_parallel is False and attr.startswith("module."):
                new_k = attr.replace("module.", "", 1)
                if '.Wqkv.' in new_k:
                    new_k = new_k.replace('.Wqkv.', '.in_proj_')
                
                new_dict[new_k] = ckpt_model[attr]
            elif data_parallel is not False and not attr.startswith("module."):
                new_dict["module." + attr] = ckpt_model[attr]
            elif data_parallel is False and not attr.startswith("module."):
                print('data_parallel is False and not attr!!!')
                new_k = attr
                if '.Wqkv.' in new_k:
                    new_k = new_k.replace('.Wqkv.', '.in_proj_')
                new_dict[new_k] = ckpt_model[attr]
            else:
                new_dict[attr] = ckpt_model[attr]
        print(new_dict.keys())
        self._load_state_dict(new_dict)
        self._load_model_weights_with_mapping(new_dict, force=force)
        print(f'load weight: {file} done!')
        return ckpt

    def _load(self, file, force=False, resume_state=False):
        ckpt = self.load_model_weights(file, force=force)

        # skip loading training state
        if resume_state is False:
            return

        if "optimizer" in ckpt:
            try:
                self.trainer.optimizer.load_state_dict(ckpt["optimizer"])
                # fix the bug of checkpoint in the pytorch with version higher than 1.11
                if "capturable" in self.trainer.optimizer.param_groups[0]:
                    self.trainer.optimizer.param_groups[0]["capturable"] = True
            except Exception as e:
                print(e)
                
        else:
            warnings.warn(
                "'optimizer' key is not present in the checkpoint asked to be loaded. Skipping."
            )

        if "lr_scheduler" in ckpt:
            self.trainer.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        else:
            warnings.warn(
                "'lr_scheduler' key is not present in the checkpoint asked to be loaded. Skipping."
            )

        self.trainer.early_stopping.init_from_checkpoint(ckpt)

        self.trainer.writer.write("Checkpoint {} loaded".format(file))

        if "current_iteration" in ckpt:
            self.trainer.current_iteration = ckpt["current_iteration"]
            registry.register("current_iteration",
                              self.trainer.current_iteration)

        if "current_epoch" in ckpt:
            self.trainer.current_epoch = ckpt["current_epoch"]
            registry.register("current_epoch", self.trainer.current_epoch)

    def save(self, iteration, update_best=False):
        if not is_main_process():
            return

        ckpt_filepath = os.path.join(self.models_foldername,
                                     "model_%d.ckpt" % iteration)
        best_ckpt_filepath = os.path.join(self.ckpt_foldername,
                                          self.ckpt_prefix + "best.ckpt")

        best_iteration = self.trainer.early_stopping.best_monitored_iteration
        best_metric = self.trainer.early_stopping.best_monitored_value
        current_iteration = self.trainer.current_iteration
        current_epoch = self.trainer.current_epoch
        model = self.trainer.model
        data_parallel = registry.get("data_parallel") or registry.get(
            "distributed")

        if data_parallel is True:
            model = model.module

        ckpt = {
            "model": model.state_dict(),
            "optimizer": self.trainer.optimizer.state_dict(),
            "lr_scheduler": self.trainer.lr_scheduler.state_dict(),
            "current_iteration": current_iteration,
            "current_epoch": current_epoch,
            "best_iteration": best_iteration,
            "best_metric_value": best_metric,
        }

        torch.save(ckpt, ckpt_filepath)
        self.remove_redundant_ckpts()

        if update_best:
            torch.save(ckpt, best_ckpt_filepath)
