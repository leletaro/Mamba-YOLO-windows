
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import mamba_ssm.ops.selective_scan_interface as ssi
from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

def _adapter_selective_scan(*args):
    """
    selective_scan_ref(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)
    """
    if len(args) == 9:
        u, delta, A, B, C, D, delta_bias, delta_softplus, return_last_state = args
        z = None
    elif len(args) == 10:
        u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state = args
    elif len(args) == 11:
        u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state, _extra = args
    else:
        raise TypeError(f"unexpected selective_scan args len={len(args)}")

    # 某些实现把 delta_bias 当 bool 开关；ref 需要张量或 None
    if isinstance(delta_bias, (bool, int)) and not isinstance(delta_bias, torch.Tensor):
        delta_bias = None

    def _f32c(t):  # 统一 dtype/布局
        return t.float().contiguous() if isinstance(t, torch.Tensor) else t

    u, delta, A, B, C, D = map(_f32c, (u, delta, A, B, C, D))
    if isinstance(z, torch.Tensor):
        z = _f32c(z)

    delta_softplus = bool(delta_softplus)
    return_last_state = bool(return_last_state)

    out = selective_scan_ref(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)

    # 关键：cross_selective_scan 期望的是 Tensor，这里做统一
    if isinstance(out, tuple):
        out = out[0]  # 丢弃 last_state
    return out

# 覆盖选择函数：避免递归 & 统一签名 & 统一返回类型
ssi.selective_scan_fn = _adapter_selective_scan
# ==== end patch ====



from ultralytics import YOLO
import argparse
import os

ROOT = os.path.abspath('.') + "/"


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT + '/ultralytics/cfg/datasets/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--config', type=str, default=ROOT + '/ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml', help='model path(s)')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--task', default='train', help='train, val, test, speed or study')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=128, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--optimizer', default='SGD', help='SGD, Adam, AdamW')
    parser.add_argument('--amp', action='store_true', help='open amp')
    parser.add_argument('--project', default=ROOT + '/output_dir/mscoco', help='save to project/name')
    parser.add_argument('--name', default='mambayolo', help='save to project/name')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    task = opt.task
    args = {
        "data": opt.data,
        "epochs": opt.epochs,
        "workers": opt.workers,
        "batch": opt.batch_size,
        "optimizer": opt.optimizer,
        "device": opt.device,
        "amp": opt.amp,
        "project": ROOT + opt.project,
        "name": opt.name,
    }
    model_conf =  r'ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml'
    task_type = {
        "train": YOLO(model_conf).train(**args),
        "val": YOLO(model_conf).val(**args),
        "test": YOLO(model_conf).test(**args),
    }
    task_type.get(task)
